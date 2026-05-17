"""Selbstspiel-Gegner: lädt das jeweils neueste Checkpoint-Modell."""

import json
import threading
import time

import numpy as np
from env import build_input, parse_obs


class ModelOpponent:
    def __init__(self, server_url: str, room: str):
        self._url = f"{server_url}?type=bot&room={room}"
        self._model = None  # wird via update_model() atomar gesetzt
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def update_model(self, model) -> None:
        with self._lock:
            self._model = model

    def start(self) -> None:
        self._stop.clear()
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        from websockets.sync.client import connect as ws_connect

        while not self._stop.is_set():
            try:
                with ws_connect(self._url, open_timeout=15) as ws:
                    self._play(ws)
            except Exception:
                if not self._stop.is_set():
                    time.sleep(1.0)

    def _play(self, ws) -> None:
        while not self._stop.is_set():
            try:
                raw = ws.recv(timeout=1.0)
            except TimeoutError:
                continue
            except Exception:
                return

            msg = json.loads(raw)
            if msg.get("type") != "observation":
                continue

            obs = parse_obs(msg)
            with self._lock:
                model = self._model

            if model is not None:
                action, _ = model.predict(obs, deterministic=False)
            else:
                action = np.random.uniform(-1.0, 1.0, size=(5,)).astype(np.float32)

            try:
                ws.send(json.dumps(build_input(action)))
            except Exception:
                return
