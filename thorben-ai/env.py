"""Gymnasium-Umgebung für einen ml-game Bot (1v1).

Observation (float32, shape=10):
  [0:2]   Vektor zum Gegner  – rel_x/2000, rel_y/2000          (0 wenn kein Gegner)
  [2:10]  Die 2 nächsten Bullets (zero-padded)
            je Bullet: rel_x/2000, rel_y/2000, cos(angle), sin(angle)

Action (float32, shape=5, alle in [-1, 1]):
  [0] up  [1] down  [2] left  [3] right  →  >0 = True
  [4] aim_angle                           →  * π = Radiant
  (shoot wird immer automatisch gesendet)
"""

import json
import math
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from websockets.sync.client import connect as ws_connect

N_BULLETS = 2
OBS_SIZE = 2 + N_BULLETS * 4  # 10

R_KILL = 0
R_DEATH = -3.0
R_STEP = 0.02  # Überleben-Bonus pro Schritt
R_DISTANCE = -0.01  # Strafe pro normierter Distanzeinheit zum Gegner
R_AIM = (
    0.05  # Bonus wenn Zielwinkel grob auf Gegner zeigt (cos-gewichtet, max pro Schritt)
)


def parse_obs(msg: dict) -> np.ndarray:
    arr = np.zeros(OBS_SIZE, dtype=np.float32)

    enemies = msg.get("enemies", [])
    if enemies:
        e = enemies[0]
        arr[0] = e["rel_x"] / 2000.0
        arr[1] = e["rel_y"] / 2000.0

    bullets = sorted(
        msg.get("bullets", []),
        key=lambda b: b["rel_x"] ** 2 + b["rel_y"] ** 2,
    )
    for i, b in enumerate(bullets[:N_BULLETS]):
        base = 2 + i * 4
        arr[base] = b["rel_x"] / 2000.0
        arr[base + 1] = b["rel_y"] / 2000.0
        arr[base + 2] = math.cos(b["angle"])
        arr[base + 3] = math.sin(b["angle"])

    return arr


def build_input(action: np.ndarray) -> dict:
    return {
        "type": "input",
        "up": bool(action[0] > 0),
        "down": bool(action[1] > 0),
        "left": bool(action[2] > 0),
        "right": bool(action[3] > 0),
        "shoot": True,
        "aim_angle": float(action[4]) * math.pi,
    }


class MlGameEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        server_url: str = "ws://localhost:3001/ws",
        room: str = "training",
        max_steps: int = 1000,
        recv_timeout: float = 5.0,
    ):
        super().__init__()
        self.server_url = server_url
        self.room = room
        self.max_steps = max_steps
        self.recv_timeout = recv_timeout

        self.observation_space = spaces.Box(-1.0, 1.0, (OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (5,), dtype=np.float32)

        self._ws = None
        self._player_id = None
        self._reward_acc = 0.0
        self._step_count = 0

    def _connect(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

        url = f"{self.server_url}?type=bot&room={self.room}"
        for attempt in range(30):
            try:
                self._ws = ws_connect(url, open_timeout=10)
                self._player_id = None
                return
            except Exception:
                time.sleep(min(2**attempt, 16))
        raise RuntimeError(f"Server nicht erreichbar: {url}")

    def _recv(self):
        try:
            return json.loads(self._ws.recv(timeout=self.recv_timeout))
        except Exception:
            self._ws = None
            return None

    def _drain_until_obs(self) -> tuple[np.ndarray, bool]:
        terminated = False
        while True:
            msg = self._recv()
            if msg is None:
                return np.zeros(OBS_SIZE, dtype=np.float32), True

            kind = msg.get("type")

            if kind == "welcome":
                self._player_id = msg["player_id"]

            elif kind == "events":
                for ev in msg.get("events", []):
                    if ev["type"] == "kill":
                        if ev.get("killer") == self._player_id:
                            self._reward_acc += R_KILL
                        if ev.get("victim") == self._player_id:
                            self._reward_acc += R_DEATH
                            terminated = True

            elif kind == "observation":
                return parse_obs(msg), terminated

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reward_acc = 0.0
        self._step_count = 0

        if self._ws is None:
            self._connect()

        # Beide Bots sofort an neuen Positionen respawnen (kein Reconnect nötig)
        try:
            self._ws.send(json.dumps({"type": "bot_reset"}))
        except Exception:
            self._ws = None
            self._connect()

        obs, _ = self._drain_until_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        self._step_count += 1
        self._reward_acc = 0.0

        try:
            self._ws.send(json.dumps(build_input(action)))
        except Exception:
            self._ws = None
            return np.zeros(OBS_SIZE, dtype=np.float32), R_DEATH, True, False, {}

        obs, terminated = self._drain_until_obs()

        # Per-step Belohnungen
        self._reward_acc += R_STEP
        if obs[0] != 0.0 or obs[1] != 0.0:  # Gegner sichtbar
            dist = math.sqrt(obs[0] ** 2 + obs[1] ** 2)
            self._reward_acc += R_DISTANCE * dist

            aim_angle = float(action[4]) * math.pi
            enemy_angle = math.atan2(obs[1], obs[0])
            alignment = math.cos(aim_angle - enemy_angle)  # 1 = perfekt, -1 = weg
            self._reward_acc += R_AIM * max(0.0, alignment)

        truncated = self._step_count >= self.max_steps
        return obs, self._reward_acc, terminated, truncated, {}

    def close(self):
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
