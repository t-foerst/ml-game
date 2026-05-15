#!/usr/bin/env python3
"""ml-game Desktop-Client.

Steuerung:
  WASD / Pfeiltasten : Bewegen (unabhaengig von Zielrichtung)
  Maus               : Zielrichtung
  LMB / Leertaste    : Schiessen
  ESC                : Zurueck ins Menue

Server-URL als optionales Argument (ueberspringt das Menue):
  python main.py ws://localhost:3001/ws
"""

import math
import sys
from typing import Optional

import pygame

from constants import BG, GRID, DARK, DIMGRAY, WHITE, GRAY, WIN_W, WIN_H, FPS, SERVERS
from network import state_q, input_q, start_network
from renderer import (
    draw_grid, draw_ship, draw_bullet, draw_effect,
    draw_minimap, draw_hud, draw_death_overlay,
    draw_enemy_indicators, draw_crosshair,
)


# ── Spielzustand (Session-weit) ───────────────────────────────────────────────
_state:  dict         = {"tick": 0, "ships": [], "bullets": []}
_my_id:  Optional[str] = None
_status: str           = "Verbinde..."


def _reset_state() -> None:
    global _my_id, _status
    _state.clear()
    _state.update({"tick": 0, "ships": [], "bullets": []})
    _my_id  = None
    _status = "Verbinde..."
    while not state_q.empty():
        state_q.get_nowait()
    while not input_q.empty():
        input_q.get_nowait()


# ── Menü ──────────────────────────────────────────────────────────────────────

def run_menu(screen: pygame.Surface, clock: pygame.time.Clock,
             font_sm: pygame.font.Font, font_lg: pygame.font.Font,
             font_xl: pygame.font.Font) -> Optional[str]:
    """Zeigt das Server-Auswahlmenü. Gibt die gewählte URL zurück oder None."""
    selected     = 0
    option_rects: list[pygame.Rect] = []

    while True:
        clock.tick(60)
        sw, sh = screen.get_size()
        cx = sw // 2

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_UP, pygame.K_w):
                    selected = (selected - 1) % len(SERVERS)
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    selected = (selected + 1) % len(SERVERS)
                elif event.key == pygame.K_RETURN:
                    return SERVERS[selected][1]
                elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return None
            elif event.type == pygame.MOUSEMOTION:
                for i, rect in enumerate(option_rects):
                    if rect.collidepoint(event.pos):
                        selected = i
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, rect in enumerate(option_rects):
                    if rect.collidepoint(event.pos):
                        return SERVERS[i][1]

        # Hintergrund + Gitter
        screen.fill(BG)
        for x in range(0, sw + 100, 100):
            pygame.draw.line(screen, GRID, (x, 0), (x, sh))
        for y in range(0, sh + 100, 100):
            pygame.draw.line(screen, GRID, (0, y), (sw, y))

        # Titel
        title = font_xl.render("ML-GAME", True, (220, 220, 220))
        screen.blit(title, (cx - title.get_width() // 2, sh // 3 - 80))

        sep_y = sh // 3 + 10
        pygame.draw.line(screen, DARK, (cx - 240, sep_y), (cx + 240, sep_y), 1)
        sub = font_sm.render("Server auswaehlen", True, GRAY)
        screen.blit(sub, (cx - sub.get_width() // 2, sep_y + 12))

        # Serveroptionen
        option_rects.clear()
        for i, (label, url) in enumerate(SERVERS):
            is_sel  = i == selected
            entry_y = sep_y + 52 + i * 80
            rect    = pygame.Rect(cx - 250, entry_y - 12, 500, 58)
            option_rects.append(rect)

            if is_sel:
                hl = pygame.Surface((500, 58), pygame.SRCALPHA)
                hl.fill((255, 255, 255, 10))
                screen.blit(hl, (cx - 250, entry_y - 12))
                pygame.draw.rect(screen, DARK, rect, 1)
                # Cursor-Dreieck
                tx, ty = cx - 238, entry_y + 10
                pygame.draw.polygon(screen, WHITE,
                                    [(tx, ty), (tx + 8, ty + 7), (tx, ty + 14)])

            col_l = WHITE if is_sel else GRAY
            col_u = GRAY  if is_sel else DIMGRAY
            screen.blit(font_lg.render(label, True, col_l), (cx - 220, entry_y))
            screen.blit(font_sm.render(url,   True, col_u),
                        (cx - 220, entry_y + font_lg.get_height() + 2))

        hint = font_sm.render(
            "hoch/runter  Auswaehlen      Enter  Verbinden      ESC  Beenden",
            True, (45, 45, 45),
        )
        screen.blit(hint, (cx - hint.get_width() // 2, sh - 44))
        pygame.display.flip()


# ── Eingabe ───────────────────────────────────────────────────────────────────

def _aim_angle(screen: pygame.Surface) -> float:
    """Zielwinkel = Winkel von Schiff-Mittelpunkt (Bildschirmmitte) zur Maus."""
    sw, sh = screen.get_size()
    mx, my = pygame.mouse.get_pos()
    return math.atan2(my - sh // 2, mx - sw // 2)


def _build_input(pressed, aim: float) -> dict:
    return {
        "type":      "input",
        "up":        bool(pressed[pygame.K_w] or pressed[pygame.K_UP]),
        "down":      bool(pressed[pygame.K_s] or pressed[pygame.K_DOWN]),
        "left":      bool(pressed[pygame.K_a] or pressed[pygame.K_LEFT]),
        "right":     bool(pressed[pygame.K_d] or pressed[pygame.K_RIGHT]),
        "shoot":     bool(pressed[pygame.K_SPACE]) or bool(pygame.mouse.get_pressed()[0]),
        "aim_angle": aim,
    }


# ── Spielschleife ─────────────────────────────────────────────────────────────

def run_game(screen: pygame.Surface, clock: pygame.time.Clock,
             server_url: str, font_sm: pygame.font.Font,
             font_md: pygame.font.Font, font_lg: pygame.font.Font,
             font_xl: pygame.font.Font) -> bool:
    """Führt die Spielschleife aus. Gibt True zurück wenn Rückkehr ins Menü."""
    global _my_id, _status

    _reset_state()
    start_network(server_url)
    pygame.display.set_caption(f"ml-game  –  {server_url}")
    pygame.mouse.set_visible(False)

    cam_x, cam_y = 0.0, 0.0
    effects:    list[dict] = []
    send_timer: float      = 0.0

    while True:
        dt = clock.tick(FPS) / 1000.0

        # ── Nachrichten vom WS-Thread ──────────────────────────────────────────
        while not state_q.empty():
            msg      = state_q.get_nowait()
            msg_type = msg.get("type")

            if msg_type == "_connected":
                _status = f"Verbunden  |  {server_url}"
            elif msg_type == "_disconnected":
                _status = "Getrennt – verbinde neu..."
                _my_id  = None
            elif msg_type == "welcome":
                _my_id = msg["player_id"]
            elif msg_type == "state":
                _state.update(msg)
            elif msg_type == "events":
                for ev in msg.get("events", []):
                    ev_type   = ev.get("type")
                    target_id = ev.get("ship") if ev_type == "hit" else ev.get("victim")
                    if not target_id:
                        continue
                    ship = next(
                        (s for s in _state.get("ships", []) if s["id"] == target_id),
                        None,
                    )
                    if ship:
                        effects.append({
                            "type": ev_type,
                            "x": ship["x"], "y": ship["y"],
                            "age": 0.0,
                            "duration": 0.7 if ev_type == "kill" else 0.25,
                        })

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return True   # zurück ins Menü

        # ── Eingabe senden (30 Hz) ─────────────────────────────────────────────
        send_timer += dt
        if send_timer >= 1.0 / 30 and _my_id:
            send_timer = 0.0
            input_q.put(_build_input(pygame.key.get_pressed(), _aim_angle(screen)))

        # ── Kamera zentriert auf eigenes Schiff ───────────────────────────────
        ships   = _state.get("ships", [])
        my_ship = next((s for s in ships if s["id"] == _my_id), None)
        if my_ship:
            cam_x, cam_y = my_ship["x"], my_ship["y"]

        # ── Zeichnen ───────────────────────────────────────────────────────────
        screen.fill(BG)
        draw_grid(screen, cam_x, cam_y)

        for b in _state.get("bullets", []):
            draw_bullet(screen, b, cam_x, cam_y)
        for s in ships:
            draw_ship(screen, s, s["id"] == _my_id, cam_x, cam_y)

        effects = [ef for ef in effects if draw_effect(screen, ef, dt, cam_x, cam_y)]
        draw_enemy_indicators(screen, ships, _my_id)

        if my_ship and not my_ship["alive"]:
            draw_death_overlay(screen, font_xl, font_md)

        draw_minimap(screen, ships, _my_id)
        draw_hud(screen, ships, _my_id, _status, font_sm, font_lg)

        mx, my_pos = pygame.mouse.get_pos()
        draw_crosshair(screen, mx, my_pos)

        pygame.display.flip()


# ── Einstiegspunkt ────────────────────────────────────────────────────────────

def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
    pygame.display.set_caption("ml-game")
    clock = pygame.time.Clock()

    font_sm = pygame.font.Font(None, 18)
    font_md = pygame.font.Font(None, 22)
    font_lg = pygame.font.Font(None, 30)
    font_xl = pygame.font.Font(None, 78)

    # CLI-Argument: Menü überspringen (z. B. für Bot-Clients)
    if len(sys.argv) > 1:
        run_game(screen, clock, sys.argv[1], font_sm, font_md, font_lg, font_xl)
        pygame.quit()
        return

    # Menü-Schleife
    while True:
        pygame.mouse.set_visible(True)
        server_url = run_menu(screen, clock, font_sm, font_lg, font_xl)
        if server_url is None:
            break
        back_to_menu = run_game(screen, clock, server_url,
                                font_sm, font_md, font_lg, font_xl)
        if not back_to_menu:
            break

    pygame.quit()


if __name__ == "__main__":
    main()
