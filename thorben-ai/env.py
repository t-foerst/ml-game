"""Gymnasium-Umgebung für einen ml-game Bot (nur Ausweichen).

Observation (float32, shape=14):
  [0:2]   Eigene Position      – x/WORLD_SIZE, y/WORLD_SIZE
  [2:4]   Gegner-Position      – x/WORLD_SIZE, y/WORLD_SIZE  (0 wenn kein Gegner)
  [4:14]  Die 5 nächsten Bullets (zero-padded)
            je Bullet: x/WORLD_SIZE, y/WORLD_SIZE

Action (float32, shape=4, alle in [-1, 1]):
  [0] up  [1] down  [2] left  [3] right  →  >0 = True
  (shoot immer False, nur Ausweichen trainiert)
"""

import json
import math
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from websockets.sync.client import connect as ws_connect

WORLD_SIZE = 1000.0  # muss mit server/game.py übereinstimmen

N_BULLETS = 5
OBS_SIZE = 2 + 2 + N_BULLETS * 2  # 14

R_DEATH = -3.0
R_STEP = 0.02  # Überleben-Bonus pro Schritt


def parse_obs(msg: dict) -> np.ndarray:
    arr = np.zeros(OBS_SIZE, dtype=np.float32)

    self_info = msg.get("self", {})
    sx = float(self_info.get("x", 0.0))
    sy = float(self_info.get("y", 0.0))
    arr[0] = sx / WORLD_SIZE
    arr[1] = sy / WORLD_SIZE

    enemies = msg.get("enemies", [])
    if enemies:
        e = enemies[0]
        arr[2] = (sx + e["rel_x"]) / WORLD_SIZE
        arr[3] = (sy + e["rel_y"]) / WORLD_SIZE

    bullets = sorted(
        msg.get("bullets", []),
        key=lambda b: b["rel_x"] ** 2 + b["rel_y"] ** 2,
    )
    for i, b in enumerate(bullets[:N_BULLETS]):
        base = 4 + i * 2
        arr[base] = (sx + b["rel_x"]) / WORLD_SIZE
        arr[base + 1] = (sy + b["rel_y"]) / WORLD_SIZE

    return arr


def build_input(action: np.ndarray) -> dict:
    return {
        "type": "input",
        "up": bool(action[0] > 0),
        "down": bool(action[1] > 0),
        "left": bool(action[2] > 0),
        "right": bool(action[3] > 0),
        "shoot": False,
        "aim_angle": 0.0,
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

        self.observation_space = spaces.Box(-2.0, 2.0, (OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)

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
                    if ev["type"] == "kill" and ev.get("victim") == self._player_id:
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

        self._reward_acc += R_STEP

        truncated = self._step_count >= self.max_steps
        return obs, self._reward_acc, terminated, truncated, {}

    def close(self):
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
