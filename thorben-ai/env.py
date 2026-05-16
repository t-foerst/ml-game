"""Gymnasium-Umgebung für einen ml-game Bot.

Observation (float32, shape=58):
  [0:6]        self  – x/2000, y/2000, sin(angle), cos(angle), health/1, cooldown/0.5
  [6:26]       enemies (4 max, zero-padded) – rel_x/2000, rel_y/2000, sin_a, cos_a, health/1
  [26:58]      bullets (8 max, zero-padded) – rel_x/2000, rel_y/2000, sin_a, cos_a

Action (float32, shape=6, all in [-1, 1]):
  [0] up    [1] down  [2] left  [3] right   → >0 = True
  [4] shoot                                 → >0 = True
  [5] aim_angle                             → * π  →  Radiant
"""

import json
import math
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from websockets.sync.client import connect as ws_connect
except ImportError:
    raise ImportError("websockets >= 12.0 benötigt:  pip install 'websockets>=12.0'")

# ── Observation-Layout ────────────────────────────────────────────────────────

N_ENEMIES  = 4
N_BULLETS  = 8
OBS_SELF   = 6
OBS_ENEMY  = 5
OBS_BULLET = 4
OBS_SIZE   = OBS_SELF + N_ENEMIES * OBS_ENEMY + N_BULLETS * OBS_BULLET  # 58

_NORM_POS = 2000.0
_NORM_HP  = 1.0
_NORM_CD  = 0.5

# ── Rewards ───────────────────────────────────────────────────────────────────

R_KILL  =  5.0
R_HIT   =  1.0   # pro Treffer auf Gegner (Kill gibt zusätzlich R_KILL)
R_DEATH = -2.0

# ── Hilfs-Funktionen (auch von opponent.py genutzt) ──────────────────────────

def parse_obs(msg: dict) -> np.ndarray:
    """Wandelt eine "observation"-Nachricht in ein normieres float32-Array um."""
    arr = np.zeros(OBS_SIZE, dtype=np.float32)
    s   = msg["self"]

    i = 0
    arr[i] = s["x"] / _NORM_POS;             i += 1
    arr[i] = s["y"] / _NORM_POS;             i += 1
    arr[i] = math.sin(s["angle"]);            i += 1
    arr[i] = math.cos(s["angle"]);            i += 1
    arr[i] = s["health"] / _NORM_HP;          i += 1
    arr[i] = s["shoot_cooldown"] / _NORM_CD;  i += 1

    for enemy in msg.get("enemies", [])[:N_ENEMIES]:
        arr[i] = enemy["rel_x"] / _NORM_POS;  i += 1
        arr[i] = enemy["rel_y"] / _NORM_POS;  i += 1
        arr[i] = math.sin(enemy["angle"]);     i += 1
        arr[i] = math.cos(enemy["angle"]);     i += 1
        arr[i] = enemy["health"] / _NORM_HP;   i += 1
    i = OBS_SELF + N_ENEMIES * OBS_ENEMY      # padding überspringen

    for bullet in msg.get("bullets", [])[:N_BULLETS]:
        arr[i] = bullet["rel_x"] / _NORM_POS; i += 1
        arr[i] = bullet["rel_y"] / _NORM_POS; i += 1
        arr[i] = math.sin(bullet["angle"]);    i += 1
        arr[i] = math.cos(bullet["angle"]);    i += 1

    return np.clip(arr, -1.0, 1.0)


def build_input(action: np.ndarray) -> dict:
    """Wandelt ein Action-Array in ein Server-Input-Dict um."""
    return {
        "type":      "input",
        "up":        bool(action[0] > 0),
        "down":      bool(action[1] > 0),
        "left":      bool(action[2] > 0),
        "right":     bool(action[3] > 0),
        "shoot":     bool(action[4] > 0),
        "aim_angle": float(action[5]) * math.pi,
    }


# ── Gymnasium-Umgebung ────────────────────────────────────────────────────────

class MlGameEnv(gym.Env):
    """Eine Gymnasium-Episode = Leben eines Bots (endet bei Tod oder max_steps)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        server_url:   str   = "ws://localhost:3001/ws",
        room:         str   = "default",
        max_steps:    int   = 2000,
        recv_timeout: float = 10.0,
    ):
        super().__init__()
        self.server_url   = server_url
        self.room         = room
        self.max_steps    = max_steps
        self.recv_timeout = recv_timeout

        self.observation_space = spaces.Box(-1.0, 1.0, (OBS_SIZE,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (6,),        dtype=np.float32)

        self._ws:         Optional[object] = None
        self._player_id:  Optional[str]    = None
        self._reward_acc: float            = 0.0
        self._step_count: int              = 0

    # ── Verbindung ────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        url = f"{self.server_url}?type=bot&room={self.room}"
        self._ws = ws_connect(url, open_timeout=15)

    # ── Nachrichtenverarbeitung ───────────────────────────────────────────────

    def _drain_until_obs(self) -> tuple[np.ndarray, bool]:
        """Liest Nachrichten bis zur nächsten Observation.
        Akkumuliert Rewards aus Events. Gibt (obs, terminated) zurück."""
        terminated = False
        while True:
            try:
                raw = self._ws.recv(timeout=self.recv_timeout)
            except Exception:
                return np.zeros(OBS_SIZE, dtype=np.float32), True

            msg  = json.loads(raw)
            kind = msg.get("type")

            if kind == "welcome":
                self._player_id = msg["player_id"]

            elif kind == "events":
                for ev in msg.get("events", []):
                    if ev["type"] == "hit":
                        if ev.get("shooter") == self._player_id:
                            self._reward_acc += R_HIT
                    elif ev["type"] == "kill":
                        if ev.get("killer") == self._player_id:
                            self._reward_acc += R_KILL
                        if ev.get("victim") == self._player_id:
                            self._reward_acc += R_DEATH
                            terminated = True

            elif kind == "observation":
                return parse_obs(msg), terminated

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._connect()
        self._reward_acc = 0.0
        self._step_count = 0
        self._player_id  = None

        obs, _ = self._drain_until_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        self._step_count += 1
        self._reward_acc  = 0.0

        try:
            self._ws.send(json.dumps(build_input(action)))
        except Exception:
            zero = np.zeros(OBS_SIZE, dtype=np.float32)
            return zero, R_DEATH, True, False, {}

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
