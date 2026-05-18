"""Algorithmischer Kreis-Gegner.

Fliegt in einer Kreisbahn um den Ursprung und wechselt periodisch zwischen:
  alpha=0  → direktes Anvisieren (aktuelle Gegnerposition)
  alpha=1  → perfekter Vorhalt   (Geschwindigkeits-Abfangberechnung)
  0<alpha<1 → zufällig dazwischen
"""

import json
import math
import random
import threading
import time

BULLET_SPEED = 620.0  # px/s (game-time) – muss mit server/game.py übereinstimmen
ORBIT_RADIUS = 600.0  # Ziel-Umlaufradius in px
AIM_MIN_INTERVAL = 1.5  # Sekunden (real) zwischen Aim-Wechseln
AIM_MAX_INTERVAL = 4.0


class CircleOpponent:
    def __init__(self, server_url: str, room: str, speed_multiplier: float = 1.0):
        self._url = f"{server_url}?type=bot&room={room}"
        self._sm = speed_multiplier  # Faktor für Bullet-Speed in real time
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
        prev_ex: float | None = None
        prev_ey: float | None = None
        prev_t:  float | None = None
        enemy_vx = 0.0
        enemy_vy = 0.0

        aim_alpha = random.random()
        next_aim_switch = time.time() + random.uniform(AIM_MIN_INTERVAL, AIM_MAX_INTERVAL)

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

            now = time.time()

            if now >= next_aim_switch:
                aim_alpha = random.random()
                next_aim_switch = now + random.uniform(AIM_MIN_INTERVAL, AIM_MAX_INTERVAL)

            self_info = msg.get("self", {})
            sx = float(self_info.get("x", 0.0))
            sy = float(self_info.get("y", 0.0))

            # ── Kreisbahn-Bewegung ────────────────────────────────────────────
            r = math.hypot(sx, sy) or 1.0
            theta = math.atan2(sy, sx)

            # Tangentialrichtung gegen den Uhrzeigersinn (Bildschirm-KS, y nach unten)
            tang_x = -math.sin(theta)
            tang_y =  math.cos(theta)

            # Radiale Korrektur: zum Zielradius drängen
            err = (r - ORBIT_RADIUS) / ORBIT_RADIUS
            corr_x = -(sx / r) * err
            corr_y = -(sy / r) * err

            move_x = tang_x + corr_x
            move_y = tang_y + corr_y

            up    = move_y < -0.2
            down  = move_y >  0.2
            left  = move_x < -0.2
            right = move_x >  0.2

            # ── Zielberechnung ───────────────────────────────────────────────
            enemies = msg.get("enemies", [])
            aim_angle = 0.0
            shoot = False

            if enemies:
                e = enemies[0]
                ex = sx + e["rel_x"]
                ey = sy + e["rel_y"]

                # Geschwindigkeitsschätzung (game-px/s, korrigiert um SPEED_MULTIPLIER)
                if prev_ex is not None and prev_t is not None:
                    real_dt = now - prev_t
                    if real_dt > 0.001:
                        enemy_vx = (ex - prev_ex) / real_dt * self._sm
                        enemy_vy = (ey - prev_ey) / real_dt * self._sm

                prev_ex, prev_ey, prev_t = ex, ey, now

                # Direktes Anvisieren
                direct_angle = math.atan2(ey - sy, ex - sx)

                # Vorhalt-Winkel (Abfangpunkt-Berechnung)
                lead_angle = direct_angle
                ddx, ddy = ex - sx, ey - sy
                d_sq    = ddx * ddx + ddy * ddy
                dot_dv  = ddx * enemy_vx + ddy * enemy_vy
                v_sq    = enemy_vx ** 2 + enemy_vy ** 2
                bs_real = BULLET_SPEED * self._sm
                bs_sq   = bs_real * bs_real

                a    = bs_sq - v_sq
                b    = -2.0 * dot_dv
                c    = -d_sq
                disc = b * b - 4.0 * a * c

                if a > 0 and disc >= 0:
                    sqrt_d = math.sqrt(disc)
                    for t_cand in ((-b - sqrt_d) / (2 * a), (-b + sqrt_d) / (2 * a)):
                        if t_cand > 0:
                            lx = ex + enemy_vx * t_cand
                            ly = ey + enemy_vy * t_cand
                            lead_angle = math.atan2(ly - sy, lx - sx)
                            break

                # Kürzeste Winkelinterpolation zwischen direkt und Vorhalt
                diff = (lead_angle - direct_angle + math.pi) % (2 * math.pi) - math.pi
                aim_angle = direct_angle + aim_alpha * diff
                shoot = True

            try:
                ws.send(json.dumps({
                    "type":      "input",
                    "up":        up,
                    "down":      down,
                    "left":      left,
                    "right":     right,
                    "shoot":     shoot,
                    "aim_angle": aim_angle,
                }))
            except Exception:
                return
