#!/usr/bin/env python3
"""client-kai — KI-Bot-Client für ml-game.

Lädt trainiertes Model aus models/ falls vorhanden.
Ohne Model: Bot steht still und kann abgeschossen werden.

Start:
  python main.py                          # Standardserver
  python main.py ws://localhost:3001/ws   # Lokaler Server
"""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import websockets

SERVERS = [
    ("Lokal",       "ws://localhost:3001/ws"),
    ("Oeffentlich", "ws://game.foerst.haus/ws"),
]
DEFAULT_SERVER = SERVERS[1][1]

MODEL_FILE = Path(__file__).parent / "models" / "ppo_model.zip"


# ── Model laden (optional) ────────────────────────────────────────────────────

def _load_model():
    if not MODEL_FILE.exists():
        print("[kai] Kein Model gefunden — Bot steht still.")
        return None
    try:
        from stable_baselines3 import PPO
        model = PPO.load(str(MODEL_FILE.with_suffix("")))
        print(f"[kai] Model geladen: {MODEL_FILE}")
        return model
    except Exception as e:
        print(f"[kai] Model-Ladefehler: {e} — Bot steht still.")
        return None


# ── KI-Entscheidung ───────────────────────────────────────────────────────────

_model     = _load_model()
_aim_angle = 0.0
_prev_enemies: list | None = None


def decide(observation: dict) -> dict | None:
    global _aim_angle, _prev_enemies

    if _model is None:
        return None

    from env import obs_to_vec, action_to_input
    obs             = obs_to_vec(observation, _prev_enemies, _aim_angle)
    _prev_enemies   = observation.get("enemies", [])
    action, _       = _model.predict(obs, deterministic=True)
    inp, _aim_angle = action_to_input(action, _aim_angle)
    return inp


# ── Bot-Schleife ──────────────────────────────────────────────────────────────

async def run(server_url: str) -> None:
    bot_url = f"{server_url}?type=bot"
    print(f"[kai] Verbinde mit {bot_url} ...")

    async with websockets.connect(bot_url) as ws:
        print("[kai] Verbunden.")
        my_id: str | None = None

        async for raw in ws:
            msg      = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "welcome":
                my_id = msg["player_id"]
                print(f"[kai] Bot-ID: {my_id}")

            elif msg_type == "observation":
                action = decide(msg)
                if action is not None:
                    await ws.send(json.dumps({"type": "input", **action}))

            elif msg_type == "events":
                for ev in msg.get("events", []):
                    if ev.get("type") == "kill" and ev.get("victim") == my_id:
                        print("[kai] Abgeschossen!")
                    elif ev.get("type") == "kill" and ev.get("killer") == my_id:
                        print("[kai] Kill!")


# ── Einstiegspunkt ────────────────────────────────────────────────────────────

def _choose_server() -> str:
    print("\nServer auswählen:")
    for i, (name, url) in enumerate(SERVERS):
        print(f"  [{i + 1}] {name}  ({url})")
    print(f"  [Enter] Standard ({DEFAULT_SERVER})")
    try:
        choice = input("> ").strip()
        idx = int(choice) - 1
        if 0 <= idx < len(SERVERS):
            return SERVERS[idx][1]
    except (ValueError, EOFError):
        pass
    return DEFAULT_SERVER


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else _choose_server()
    try:
        asyncio.run(run(url))
    except KeyboardInterrupt:
        print("\n[kai] Bot beendet.")


if __name__ == "__main__":
    main()
