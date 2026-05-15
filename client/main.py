#!/usr/bin/env python3
"""Panzer-Spiel Desktop-Client (pygame).

Steuerung:
  WASD / Pfeiltasten : Fahren / Drehen
  Leertaste          : Schießen
  ESC / Q            : Beenden

Server-URL als optionales Argument:
  python main.py ws://localhost:3001/ws
"""

import asyncio
import json
import math
import queue
import sys
import threading
from typing import Optional

import pygame
import websockets

# ── Konfiguration ──────────────────────────────────────────────────────────────
SERVER_URL    = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:3001/ws"
WIN_W, WIN_H  = 1280, 720
FPS           = 60
TANK_MAX_HP   = 3
MINIMAP_SIZE  = 160
MINIMAP_SCALE = 1 / 20      # 1 Minimap-px = 20 Welteinheiten

# ── Farben (schwarz-weiß Palette) ─────────────────────────────────────────────
BG      = (17,  17,  17)
GRID    = (28,  28,  28)
ORIGIN  = (40,  40,  40)
WHITE   = (230, 230, 230)
GRAY    = (120, 120, 120)
DARK    = (65,  65,  65)
DIMGRAY = (40,  40,  40)
HP_HIGH = (190, 190, 190)
HP_MID  = (110, 110, 110)
HP_LOW  = (65,  65,  65)

# ── Thread-sichere Queues ─────────────────────────────────────────────────────
state_q: queue.Queue = queue.Queue()    # WS-Thread → Main-Thread
input_q: queue.Queue = queue.Queue()    # Main-Thread → WS-Thread

# Geteilter Spielzustand (nur Main-Thread schreibt/liest nach Empfang)
game_state: dict = {"tick": 0, "tanks": [], "bullets": []}
my_id:      Optional[str] = None
conn_status: str           = "Verbinde..."


# ── WebSocket-Hintergrundthread ────────────────────────────────────────────────

async def _recv(ws) -> None:
    async for raw in ws:
        state_q.put(json.loads(raw))


async def _send_loop(ws) -> None:
    while True:
        try:
            msg = input_q.get_nowait()
            await ws.send(json.dumps(msg))
        except queue.Empty:
            pass
        await asyncio.sleep(1.0 / 60)


async def _ws_run() -> None:
    while True:
        try:
            async with websockets.connect(SERVER_URL) as ws:
                state_q.put({"type": "_connected"})
                recv_task = asyncio.create_task(_recv(ws))
                send_task = asyncio.create_task(_send_loop(ws))
                _done, pending = await asyncio.wait(
                    [recv_task, send_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass
        except Exception:
            pass
        state_q.put({"type": "_disconnected"})
        await asyncio.sleep(2.0)


def _ws_thread() -> None:
    asyncio.run(_ws_run())


# ── Geometrie-Hilfen ──────────────────────────────────────────────────────────

def w2s(wx: float, wy: float, cam_x: float, cam_y: float, sw: int, sh: int) -> tuple[int, int]:
    """Welt- → Bildschirmkoordinaten."""
    return (int(wx - cam_x + sw // 2), int(wy - cam_y + sh // 2))


def rotate_pts(pts: list[tuple[float, float]], angle: float, ox: float, oy: float) -> list[tuple[float, float]]:
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return [
        (ox + px * cos_a - py * sin_a,
         oy + px * sin_a + py * cos_a)
        for px, py in pts
    ]


# ── Zeichenfunktionen ─────────────────────────────────────────────────────────

def draw_grid(surf: pygame.Surface, cam_x: float, cam_y: float) -> None:
    sw, sh = surf.get_size()
    gs = 100
    x0 = math.floor((cam_x - sw / 2) / gs) * gs
    y0 = math.floor((cam_y - sh / 2) / gs) * gs

    for wx in range(int(x0), int(cam_x + sw / 2 + gs), gs):
        sx = int(wx - cam_x + sw // 2)
        pygame.draw.line(surf, GRID, (sx, 0), (sx, sh))
    for wy in range(int(y0), int(cam_y + sh / 2 + gs), gs):
        sy = int(wy - cam_y + sh // 2)
        pygame.draw.line(surf, GRID, (0, sy), (sw, sy))

    # Ursprungsmarkierung
    ox, oy = w2s(0, 0, cam_x, cam_y, sw, sh)
    pygame.draw.line(surf, ORIGIN, (ox - 12, oy), (ox + 12, oy), 2)
    pygame.draw.line(surf, ORIGIN, (ox, oy - 12), (ox, oy + 12), 2)


def draw_tank(surf: pygame.Surface, tank: dict, is_me: bool, cam_x: float, cam_y: float) -> None:
    if not tank["alive"]:
        return

    sw, sh = surf.get_size()
    sx, sy = w2s(tank["x"], tank["y"], cam_x, cam_y, sw, sh)
    angle  = tank["angle"]
    fill   = WHITE if is_me else GRAY
    stroke = DARK  if is_me else DIMGRAY

    def rot(pts):
        return rotate_pts(pts, angle, sx, sy)

    # Rumpf
    body = rot([(-17, -13), (17, -13), (17, 13), (-17, 13)])
    pygame.draw.polygon(surf, fill,   body)
    pygame.draw.polygon(surf, stroke, body, 2)

    # Lauf
    barrel = rot([(0, -4), (22, -4), (22, 4), (0, 4)])
    pygame.draw.polygon(surf, fill,   barrel)
    pygame.draw.polygon(surf, stroke, barrel, 2)

    # Turm
    pygame.draw.circle(surf, fill,   (sx, sy), 7)
    pygame.draw.circle(surf, stroke, (sx, sy), 7, 2)

    # Lebensbalken
    bw, bh = 38, 4
    bx, by = sx - bw // 2, sy - 32
    pygame.draw.rect(surf, (30, 30, 30), (bx, by, bw, bh))
    ratio   = tank["health"] / TANK_MAX_HP
    bar_col = HP_HIGH if ratio > 0.6 else (HP_MID if ratio > 0.3 else HP_LOW)
    pygame.draw.rect(surf, bar_col, (bx, by, int(bw * ratio), bh))


def draw_bullet(surf: pygame.Surface, b: dict, cam_x: float, cam_y: float) -> None:
    sw, sh = surf.get_size()
    sx, sy = w2s(b["x"], b["y"], cam_x, cam_y, sw, sh)
    # Glüh-Halo
    glow = pygame.Surface((18, 18), pygame.SRCALPHA)
    pygame.draw.circle(glow, (255, 255, 255, 30), (9, 9), 8)
    surf.blit(glow, (sx - 9, sy - 9))
    pygame.draw.circle(surf, WHITE, (sx, sy), 4)


def draw_effect(surf: pygame.Surface, ef: dict, dt: float, cam_x: float, cam_y: float) -> bool:
    """Zeichnet visuellen Effekt; gibt False zurück wenn abgelaufen."""
    ef["age"] += dt
    if ef["age"] >= ef["duration"]:
        return False

    sw, sh = surf.get_size()
    sx, sy = w2s(ef["x"], ef["y"], cam_x, cam_y, sw, sh)
    p     = ef["age"] / ef["duration"]
    alpha = int((1 - p) * 220)

    if ef["type"] == "kill":
        for radius, a_factor in [(int(60 * p), 1.0), (int(32 * p), 0.4)]:
            if radius < 1:
                continue
            tmp = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(tmp, (210, 210, 210, int(alpha * a_factor)),
                               (radius + 2, radius + 2), radius, 3)
            surf.blit(tmp, (sx - radius - 2, sy - radius - 2))
    else:
        r = int(24 * p)
        if r > 0:
            tmp = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(tmp, (200, 200, 200, alpha), (r + 2, r + 2), r, 2)
            surf.blit(tmp, (sx - r - 2, sy - r - 2))

    return True


def draw_minimap(surf: pygame.Surface, tanks: list, my_id_: Optional[str]) -> None:
    S = MINIMAP_SIZE
    mm = pygame.Surface((S, S), pygame.SRCALPHA)
    mm.fill((0, 0, 0, 160))
    pygame.draw.rect(mm, (42, 42, 42, 255), mm.get_rect(), 1)

    my_tank = next((t for t in tanks if t["id"] == my_id_), None)
    cx = my_tank["x"] if my_tank else 0.0
    cy = my_tank["y"] if my_tank else 0.0

    for t in tanks:
        if not t["alive"]:
            continue
        mx = int(S / 2 + (t["x"] - cx) * MINIMAP_SCALE)
        my_ = int(S / 2 + (t["y"] - cy) * MINIMAP_SCALE)
        if 0 <= mx < S and 0 <= my_ < S:
            col = (230, 230, 230) if t["id"] == my_id_ else (90, 90, 90)
            r   = 4 if t["id"] == my_id_ else 3
            pygame.draw.circle(mm, col, (mx, my_), r)

    half = S // 2
    pygame.draw.line(mm, (40, 40, 40), (half - 6, half), (half + 6, half))
    pygame.draw.line(mm, (40, 40, 40), (half, half - 6), (half, half + 6))

    sw, sh = surf.get_size()
    surf.blit(mm, (sw - S - 16, sh - S - 16))


def draw_hud(surf: pygame.Surface, tanks: list, my_id_: Optional[str],
             status: str, font_sm: pygame.font.Font, font_lg: pygame.font.Font) -> None:
    surf_w = surf.get_width()

    # Verbindungsstatus
    s = font_sm.render(status, True, (70, 70, 70))
    surf.blit(s, (16, 16))

    # Lebensanzeige
    my_tank = next((t for t in tanks if t["id"] == my_id_), None)
    if my_tank:
        hp_text = "█" * my_tank["health"] + "░" * (TANK_MAX_HP - my_tank["health"])
        s = font_lg.render(hp_text, True, WHITE)
        surf.blit(s, (16, 36))

    # Steuerungshinweis
    hint = font_sm.render("WASD / Pfeile: Fahren  ·  Leertaste: Schiessen  ·  ESC: Beenden", True, (50, 50, 50))
    surf.blit(hint, (16, 70))

    # Punktestand
    title = font_sm.render("PUNKTE", True, (48, 48, 48))
    surf.blit(title, (surf_w - title.get_width() - 16, 16))

    for i, t in enumerate(sorted(tanks, key=lambda x: -x["score"])):
        you  = "  <- du" if t["id"] == my_id_ else ""
        dead = "  x"    if not t["alive"]        else ""
        col  = (200, 200, 200) if t["id"] == my_id_ else (90, 90, 90)
        label = font_sm.render(f"{t['score']} kills{you}{dead}", True, col)
        surf.blit(label, (surf_w - label.get_width() - 16, 36 + i * 20))


def draw_death_overlay(surf: pygame.Surface,
                       font_xl: pygame.font.Font, font_md: pygame.font.Font) -> None:
    overlay = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 140))
    surf.blit(overlay, (0, 0))

    sw, sh = surf.get_size()
    t1 = font_xl.render("ZERSTOERT", True, (190, 190, 190))
    t2 = font_md.render("Respawn in 3 Sekunden...", True, (75, 75, 75))
    surf.blit(t1, (sw // 2 - t1.get_width() // 2, sh // 2 - t1.get_height() // 2))
    surf.blit(t2, (sw // 2 - t2.get_width() // 2, sh // 2 + 44))


# ── Eingabe ───────────────────────────────────────────────────────────────────

def build_input(pressed) -> dict:
    return {
        "type":     "input",
        "forward":  bool(pressed[pygame.K_w] or pressed[pygame.K_UP]),
        "backward": bool(pressed[pygame.K_s] or pressed[pygame.K_DOWN]),
        "left":     bool(pressed[pygame.K_a] or pressed[pygame.K_LEFT]),
        "right":    bool(pressed[pygame.K_d] or pressed[pygame.K_RIGHT]),
        "shoot":    bool(pressed[pygame.K_SPACE]),
    }


# ── Hauptschleife ─────────────────────────────────────────────────────────────

def main() -> None:
    global game_state, my_id, conn_status

    threading.Thread(target=_ws_thread, daemon=True).start()

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
    pygame.display.set_caption("Tank Game")
    clock = pygame.time.Clock()

    font_sm = pygame.font.Font(None, 18)
    font_md = pygame.font.Font(None, 22)
    font_lg = pygame.font.Font(None, 30)
    font_xl = pygame.font.Font(None, 78)

    cam_x, cam_y = 0.0, 0.0
    effects: list[dict] = []
    send_timer = 0.0

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        # ── Nachrichten vom WS-Thread verarbeiten ──────────────────────────────
        while not state_q.empty():
            msg = state_q.get_nowait()
            msg_type = msg.get("type")

            if msg_type == "_connected":
                conn_status = "Verbunden"
            elif msg_type == "_disconnected":
                conn_status = "Getrennt - verbinde neu..."
                my_id = None
            elif msg_type == "welcome":
                my_id = msg["player_id"]
            elif msg_type == "state":
                game_state = msg
            elif msg_type == "events":
                for ev in msg.get("events", []):
                    ev_type = ev.get("type")
                    if ev_type == "hit":
                        target_id = ev.get("tank")
                    elif ev_type == "kill":
                        target_id = ev.get("victim")
                    else:
                        continue
                    tank = next((t for t in game_state["tanks"] if t["id"] == target_id), None)
                    if tank:
                        effects.append({
                            "type": ev_type,
                            "x": tank["x"], "y": tank["y"],
                            "age": 0.0,
                            "duration": 0.7 if ev_type == "kill" else 0.25,
                        })

        # ── pygame-Events ──────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

        # ── Eingabe senden (30 Hz) ─────────────────────────────────────────────
        send_timer += dt
        if send_timer >= 1.0 / 30 and my_id:
            send_timer = 0.0
            input_q.put(build_input(pygame.key.get_pressed()))

        # ── Kamera zentriert auf eigenen Panzer ────────────────────────────────
        tanks   = game_state.get("tanks", [])
        my_tank = next((t for t in tanks if t["id"] == my_id), None)
        if my_tank:
            cam_x, cam_y = my_tank["x"], my_tank["y"]

        # ── Zeichnen ───────────────────────────────────────────────────────────
        screen.fill(BG)
        draw_grid(screen, cam_x, cam_y)

        for b in game_state.get("bullets", []):
            draw_bullet(screen, b, cam_x, cam_y)
        for t in tanks:
            draw_tank(screen, t, t["id"] == my_id, cam_x, cam_y)

        effects = [ef for ef in effects if draw_effect(screen, ef, dt, cam_x, cam_y)]

        if my_tank and not my_tank["alive"]:
            draw_death_overlay(screen, font_xl, font_md)

        draw_minimap(screen, tanks, my_id)
        draw_hud(screen, tanks, my_id, conn_status, font_sm, font_lg)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
