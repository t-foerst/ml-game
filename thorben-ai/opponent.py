"""Algorithmischer Gegner-Bot: stationär, schießt immer direkt auf den Gegner."""

import json
import math
import threading
import time


class AlgorithmicBot:
    def __init__(self, server_url: str, room: str):
        self._url  = f"{server_url}?type=bot&room={room}"
        self._stop = threading.Event()

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
        my_id = None

        while not self._stop.is_set():
            try:
                raw = ws.recv(timeout=1.0)
            except TimeoutError:
                continue
            except Exception:
                return

            msg  = json.loads(raw)
            kind = msg.get("type")

            if kind == "welcome":
                my_id = msg.get("player_id")

            elif kind == "observation":
                enemies = msg.get("enemies", [])
                if enemies:
                    e   = enemies[0]
                    aim = math.atan2(e["rel_y"], e["rel_x"])
                else:
                    aim = 0.0

                try:
                    ws.send(json.dumps({
                        "type":      "input",
                        "up":        False,
                        "down":      False,
                        "left":      False,
                        "right":     False,
                        "shoot":     bool(enemies),
                        "aim_angle": aim,
                    }))
                except Exception:
                    return
