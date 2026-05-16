"""Gegner-Bot für Selbstspiel.

Läuft in einem Daemon-Thread, verbindet sich mit dem Server als Bot und
spielt mit der aktuell geladenen Policy. Wird vom SelfPlayCallback periodisch
auf einen neuen Checkpoint aktualisiert.
"""

import json
import threading
import time
from typing import Optional

import numpy as np

from env import parse_obs, build_input, OBS_SIZE


def _random_action() -> np.ndarray:
    return np.random.uniform(-1.0, 1.0, size=(6,)).astype(np.float32)


class OpponentBot:
    """Selbstspiel-Gegner: verbindet sich als Bot und agiert mit einer Policy."""

    def __init__(self, server_url: str, room: str):
        self._url    = f"{server_url}?type=bot&room={room}"
        self._model  = None          # SB3-Modell, thread-safe via Lock
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── Öffentliche API ───────────────────────────────────────────────────────

    def update_model(self, model) -> None:
        """Setzt das Modell atomar (thread-safe). Darf jederzeit aufgerufen werden."""
        with self._lock:
            self._model = model

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"opp:{self._url}"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ── Bot-Loop ──────────────────────────────────────────────────────────────

    def _run(self) -> None:
        from websockets.sync.client import connect as ws_connect

        while not self._stop.is_set():
            try:
                with ws_connect(self._url, open_timeout=15) as ws:
                    self._play(ws)
            except Exception:
                if not self._stop.is_set():
                    time.sleep(2.0)

    def _play(self, ws) -> None:
        while not self._stop.is_set():
            try:
                raw = ws.recv(timeout=1.0)
            except TimeoutError:
                continue
            except Exception:
                return

            msg  = json.loads(raw)
            kind = msg.get("type")

            if kind == "observation":
                obs = parse_obs(msg)
                with self._lock:
                    model = self._model
                action = (
                    model.predict(obs, deterministic=False)[0]
                    if model is not None
                    else _random_action()
                )
                try:
                    ws.send(json.dumps(build_input(action)))
                except Exception:
                    return
