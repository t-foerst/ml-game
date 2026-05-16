"""live.py — KI-Bot der auf dem echten Server spielt UND dabei lernt.

Der Bot verbindet sich als ?type=bot, empfängt Observations in Echtzeit,
entscheidet per PPO-Model und sammelt Erfahrungen für Online-Training.
Alle N Steps wird das Model aktualisiert und gespeichert.

Offline-Vortraining (schnell):   python agent.py
Online-Feintraining (Echtzeit):  python live.py
Beide teilen dasselbe Model in models/.

Start:
  python live.py
  python live.py ws://localhost:3001/ws
"""

import asyncio
import json
import queue
import sys
import threading
from pathlib import Path

import numpy as np
import websockets
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces

from env import obs_to_vec, action_to_input, OBS_SIZE
from agent import load, save, DEFAULT_CONFIG

SERVERS = [
    ("Lokal",       "ws://localhost:3001/ws"),
    ("Oeffentlich", "ws://game.foerst.haus/ws"),
]

# Für Live-Training kleinere Batches — Update alle ~13 Sekunden (256 / 20Hz)
LIVE_N_STEPS  = 256
SAVE_EVERY    = LIVE_N_STEPS * 4   # alle 4 Updates speichern

from env import R_HIT_LANDED, R_HIT_TAKEN


# ── Live-Environment (WebSocket ↔ Gymnasium) ──────────────────────────────────

class LiveGameEnv(gym.Env):
    """Gymnasium-Environment das den echten Server per WebSocket anspricht.

    step() blockiert bis die nächste Observation vom Server eintrifft.
    Der WebSocket läuft in einem eigenen Thread.
    """
    metadata = {"render_modes": []}

    def __init__(self, server_url: str):
        super().__init__()
        self.action_space      = spaces.MultiDiscrete([2, 2, 2, 2, 2, 3])
        self.observation_space = spaces.Box(-1., 1., (OBS_SIZE,), np.float32)

        self._server_url = server_url
        self._obs_q        = queue.Queue()
        self._action_q     = queue.Queue()
        self._reward_acc   = 0.0
        self._episode_done = False
        self._my_id        = None
        self._aim_angle    = 0.0
        self._prev_enemies = None

        self._loop   = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_ws, daemon=True)
        self._thread.start()

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reward_acc   = 0.0
        self._episode_done = False
        self._aim_angle    = 0.0
        # Leere alte Observations (z.B. nach Respawn)
        while not self._obs_q.empty():
            try:
                self._obs_q.get_nowait()
            except queue.Empty:
                break
        obs = self._obs_q.get(timeout=30)
        return obs, {}

    def step(self, action):
        inp, self._aim_angle = action_to_input(action, self._aim_angle)
        self._action_q.put(inp)

        try:
            obs = self._obs_q.get(timeout=5)
        except queue.Empty:
            print("[live] Timeout — Verbindung verloren?")
            return np.zeros(OBS_SIZE, np.float32), 0.0, True, False, {}

        reward             = self._reward_acc
        self._reward_acc   = 0.0
        done               = self._episode_done
        self._episode_done = False
        return obs, reward, done, False, {}

    # ── WebSocket-Thread ───────────────────────────────────────────────────────

    def _run_ws(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._ws_loop())

    async def _ws_loop(self):
        bot_url = f"{self._server_url}?type=bot"
        while True:
            try:
                async with websockets.connect(bot_url) as ws:
                    print(f"[live] Verbunden — {bot_url}")
                    sender = asyncio.create_task(self._sender(ws))
                    async for raw in ws:
                        self._handle(json.loads(raw))
                    sender.cancel()
            except Exception as e:
                print(f"[live] Verbindungsfehler: {e} — verbinde neu...")
                await asyncio.sleep(2)

    async def _sender(self, ws):
        loop = asyncio.get_running_loop()
        while True:
            try:
                inp = await loop.run_in_executor(
                    None, self._action_q.get, True, 1.0)
                await ws.send(json.dumps({"type": "input", **inp}))
            except (queue.Empty, TimeoutError):
                pass
            except Exception:
                break

    def _handle(self, msg: dict):
        mtype = msg.get("type")
        if mtype == "welcome":
            self._my_id = msg["player_id"]
            print(f"[live] Bot-ID: {self._my_id}")
        elif mtype == "observation":
            server_angle = msg.get("self", {}).get("angle", self._aim_angle)
            self._obs_q.put(obs_to_vec(msg, self._prev_enemies, server_angle))
            self._prev_enemies = msg.get("enemies", [])
        elif mtype == "events":
            for ev in msg.get("events", []):
                if ev["type"] == "kill":
                    if ev.get("victim") == self._my_id:
                        self._reward_acc   += R_HIT_TAKEN
                        self._episode_done  = True
                        print("[live] Abgeschossen!")
                    elif ev.get("killer") == self._my_id:
                        self._reward_acc += R_HIT_LANDED
                        print("[live] Kill!")
                elif ev["type"] == "hit" and ev.get("ship") == self._my_id:
                    self._reward_acc += R_HIT_TAKEN


# ── Callback: Speichern ───────────────────────────────────────────────────────

class _LiveSaveCallback(BaseCallback):
    def __init__(self, config: dict, save_every: int):
        super().__init__()
        self._config     = config
        self._save_every = save_every
        self._next_save  = save_every

    def _on_step(self) -> bool:
        if self.model.num_timesteps >= self._next_save:
            self._next_save += self._save_every
            save(self.model, self._config)
        return True


# ── Training ──────────────────────────────────────────────────────────────────

def train_live(server_url: str) -> None:
    model, config = load()
    env = LiveGameEnv(server_url)

    live_config = {**config, "n_steps": LIVE_N_STEPS}

    if model is None:
        model = PPO(
            "MlpPolicy", env,
            learning_rate = live_config["learning_rate"],
            n_steps       = LIVE_N_STEPS,
            batch_size    = min(live_config["batch_size"], LIVE_N_STEPS),
            n_epochs      = live_config["n_epochs"],
            gamma         = live_config["gamma"],
            gae_lambda    = live_config["gae_lambda"],
            clip_range    = live_config["clip_range"],
            ent_coef      = live_config["ent_coef"],
            verbose       = 1,
        )
    else:
        model.set_env(env)
        print(f"[live] Fortsetzung ab {config['total_timesteps_trained']:,} Steps")

    callback = _LiveSaveCallback(config, SAVE_EVERY)

    print("[live] Training läuft — Strg+C zum Beenden")
    try:
        model.learn(
            total_timesteps     = 10_000_000,  # läuft bis Strg+C
            callback            = callback,
            reset_num_timesteps = False,
        )
    except KeyboardInterrupt:
        print("\n[live] Unterbrochen —", end=" ")

    save(model, config)


# ── Nur spielen (kein Training) ───────────────────────────────────────────────

async def _play_loop(server_url: str, model) -> None:
    bot_url = f"{server_url}?type=bot"
    prev_enemies = None
    my_id        = None

    while True:
        try:
            async with websockets.connect(bot_url) as ws:
                print(f"[live] Verbunden (kein Training) — {bot_url}")
                async for raw in ws:
                    msg   = json.loads(raw)
                    mtype = msg.get("type")
                    if mtype == "welcome":
                        my_id = msg["player_id"]
                        print(f"[live] Bot-ID: {my_id}")
                    elif mtype == "observation":
                        # Server-Winkel als Ground-Truth — kein lokales Tracking
                        aim_angle    = msg.get("self", {}).get("angle", 0.0)
                        obs          = obs_to_vec(msg, prev_enemies, aim_angle)
                        prev_enemies = msg.get("enemies", [])
                        action, _    = model.predict(obs, deterministic=False)
                        inp, _       = action_to_input(action, aim_angle)
                        await ws.send(json.dumps({"type": "input", **inp}))
                    elif mtype == "events":
                        for ev in msg.get("events", []):
                            if ev.get("type") == "kill":
                                if ev.get("victim") == my_id:
                                    print("[live] Abgeschossen!")
                                    prev_enemies = None
                                elif ev.get("killer") == my_id:
                                    print("[live] Kill!")
        except KeyboardInterrupt:
            return
        except Exception as e:
            print(f"[live] Verbindungsfehler: {e} — verbinde neu...")
            await asyncio.sleep(2)


def play_live(server_url: str) -> None:
    model, config = load()
    if model is None:
        print("[live] Kein Model gefunden — zuerst trainieren (agent.py).")
        return
    print(f"[live] Spiele ohne Training — {config['total_timesteps_trained']:,} Steps trainiert.")
    try:
        asyncio.run(_play_loop(server_url, model))
    except KeyboardInterrupt:
        print("\n[live] Beendet.")


# ── Einstiegspunkt ────────────────────────────────────────────────────────────

MENU = [
    ("Oeffentlicher Server",              SERVERS[1][1], False),
    ("Lokaler Server",                    SERVERS[0][1], False),
    ("Oeffentlicher Server + Training",   SERVERS[1][1], True),
    ("Lokaler Server    + Training",      SERVERS[0][1], True),
]


def _choose() -> tuple[str, bool]:
    print("\nModus auswählen:")
    for i, (label, url, train) in enumerate(MENU):
        suffix = f"  ({url})"
        print(f"  [{i + 1}] {label}{suffix}")
    while True:
        try:
            idx = int(input("> ").strip()) - 1
            if 0 <= idx < len(MENU):
                _, url, train = MENU[idx]
                return url, train
        except (ValueError, EOFError):
            pass
        print(f"  Bitte 1–{len(MENU)} eingeben.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        url, do_train = sys.argv[1], False
    else:
        url, do_train = _choose()

    if do_train:
        train_live(url)
    else:
        play_live(url)
