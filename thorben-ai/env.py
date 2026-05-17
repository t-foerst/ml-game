"""Gymnasium-Umgebung für einen ml-game Bot (1v1).

Observation (float32, shape=10):
  [0:2]   Vektor zum Gegner  – rel_x/2000, rel_y/2000          (0 wenn kein Gegner)
  [2:10]  Die 2 nächsten Bullets (zero-padded)
            je Bullet: rel_x/2000, rel_y/2000, cos(angle), sin(angle)

Action (float32, shape=6, alle in [-1, 1]):
  [0] up  [1] down  [2] left  [3] right  →  >0 = True
  [4] shoot                               →  >0 = True
  [5] aim_angle                           →  * π = Radiant
"""

import json
import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from websockets.sync.client import connect as ws_connect

N_BULLETS = 2
OBS_SIZE  = 2 + N_BULLETS * 4  # 10

R_KILL  =  5.0
R_DEATH = -2.0


def parse_obs(msg: dict) -> np.ndarray:
    arr = np.zeros(OBS_SIZE, dtype=np.float32)

    enemies = msg.get("enemies", [])
    if enemies:
        e      = enemies[0]
        arr[0] = e["rel_x"] / 2000.0
        arr[1] = e["rel_y"] / 2000.0

    bullets = sorted(
        msg.get("bullets", []),
        key=lambda b: b["rel_x"] ** 2 + b["rel_y"] ** 2,
    )
    for i, b in enumerate(bullets[:N_BULLETS]):
        base          = 2 + i * 4
        arr[base]     = b["rel_x"] / 2000.0
        arr[base + 1] = b["rel_y"] / 2000.0
        arr[base + 2] = math.cos(b["angle"])
        arr[base + 3] = math.sin(b["angle"])

    return arr


def build_input(action: np.ndarray) -> dict:
    return {
        "type":      "input",
        "up":        bool(action[0] > 0),
        "down":      bool(action[1] > 0),
        "left":      bool(action[2] > 0),
        "right":     bool(action[3] > 0),
        "shoot":     bool(action[4] > 0),
        "aim_angle": float(action[5]) * math.pi,
    }


class MlGameEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        server_url:   str   = "ws://localhost:3001/ws",
        room:         str   = "training",
        max_steps:    int   = 300,
        recv_timeout: float = 5.0,
    ):
        super().__init__()
        self.server_url   = server_url
        self.room         = room
        self.max_steps    = max_steps
        self.recv_timeout = recv_timeout

        self.observation_space = spaces.Box(-1.0, 1.0, (OBS_SIZE,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (6,),        dtype=np.float32)

        self._ws         = None
        self._player_id  = None
        self._reward_acc = 0.0
        self._step_count = 0
        self._dead       = False

    def _connect(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        self._ws        = ws_connect(
            f"{self.server_url}?type=bot&room={self.room}",
            open_timeout=10,
        )
        self._player_id = None
        self._dead      = False

    def _recv(self):
        """Nächste Nachricht lesen. Bei Fehler → ws auf None setzen."""
        try:
            return json.loads(self._ws.recv(timeout=self.recv_timeout))
        except Exception:
            self._ws = None
            return None

    def _drain_until_obs(self) -> tuple[np.ndarray, bool]:
        """Liest bis zur nächsten Observation, akkumuliert Rewards."""
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
                            self._dead = True

            elif kind == "observation":
                return parse_obs(msg), terminated

    def _send_bot_reset(self) -> None:
        """Fordert sofortiges Respawn beider Bots an neuen Positionen an."""
        if self._ws is None:
            return
        try:
            self._ws.send(json.dumps({"type": "bot_reset"}))
        except Exception:
            self._ws = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reward_acc = 0.0
        self._step_count = 0
        self._dead = False

        if self._ws is None:
            self._connect()

        # Bot-Reset: Tod-Timer umgehen, beide Bots sofort neu positionieren
        self._send_bot_reset()

        obs, _ = self._drain_until_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        self._step_count += 1
        self._reward_acc  = 0.0

        try:
            self._ws.send(json.dumps(build_input(action)))
        except Exception:
            self._ws = None
            return np.zeros(OBS_SIZE, dtype=np.float32), R_DEATH, True, False, {}

        obs, terminated = self._drain_until_obs()
        truncated = self._step_count >= self.max_steps
        return obs, self._reward_acc, terminated, truncated, {}

    def close(self):
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
