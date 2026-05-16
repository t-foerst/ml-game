#!/usr/bin/env python3
"""ml-game Desktop-Client.

Steuerung (Spieler):
  WASD / Pfeiltasten : Bewegen (unabhaengig von Zielrichtung)
  Maus               : Zielrichtung
  LMB / Leertaste    : Schiessen
  ESC                : Zurueck ins Menue

Steuerung (Zuschauer):
  WASD / Pfeiltasten : Kamera verschieben
  ESC                : Zurueck ins Menue

Server-URL als optionales Argument (ueberspringt das Menue):
  python main.py ws://localhost:3001/ws
  python main.py ws://localhost:3001/ws myroom
  python main.py ws://localhost:3001/ws myroom spectate
"""

import json
import math
import sys
import threading
import urllib.request
from typing import Optional
from urllib.parse import urlparse, urlunparse

import pygame

from constants import BG, GRID, DARK, DIMGRAY, WHITE, GRAY, WIN_W, WIN_H, FPS, SERVERS
from network import state_q, input_q, start_network, stop_network
from renderer import (
    draw_background, draw_nebula,
    draw_grid, draw_ship, draw_bullet, draw_effect,
    draw_minimap, draw_hud, draw_death_overlay,
    draw_enemy_indicators, draw_crosshair,
)
from sound import SoundManager

SPECTATOR_CAM_SPEED = 500.0
SHOOT_COOLDOWN      = 0.5    # muss mit server/game.py übereinstimmen

# ── Spielzustand (Session-weit) ───────────────────────────────────────────────
_state:  dict          = {"tick": 0, "ships": [], "bullets": []}
_my_id:  Optional[str] = None
_status: str           = "Verbinde..."


def _reset_state() -> None:
    global _my_id, _status
    stop_network()
    _state.clear()
    _state.update({"tick": 0, "ships": [], "bullets": []})
    _my_id  = None
    _status = "Verbinde..."
    while not state_q.empty():
        state_q.get_nowait()
    while not input_q.empty():
        input_q.get_nowait()


# ── HTTP-Hilfen ───────────────────────────────────────────────────────────────

def _rooms_http_url(server_url: str) -> str:
    """ws://host:port/ws  →  http://host:port/rooms"""
    p = urlparse(server_url)
    scheme = "https" if p.scheme == "wss" else "http"
    return urlunparse(p._replace(scheme=scheme, path="/rooms", query="", fragment=""))


def _fetch_rooms_async(server_url: str, result: dict) -> None:
    try:
        url = _rooms_http_url(server_url)
        with urllib.request.urlopen(url, timeout=3) as r:
            result["data"] = json.loads(r.read())
    except Exception as e:
        result["error"] = str(e)
    finally:
        result["done"] = True


# ── Menü: Server-Auswahl ──────────────────────────────────────────────────────

def run_server_menu(screen: pygame.Surface, clock: pygame.time.Clock,
                    font_sm: pygame.font.Font, font_lg: pygame.font.Font,
                    font_xl: pygame.font.Font,
                    sound_enabled: bool) -> tuple[Optional[str], bool]:
    """Gibt (Server-URL oder None, sound_enabled) zurück."""
    selected      = 0
    option_rects: list[pygame.Rect] = []
    checkbox_rect = pygame.Rect(0, 0, 14, 14)

    while True:
        clock.tick(60)
        sw, sh = screen.get_size()
        cx = sw // 2

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, sound_enabled
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_UP, pygame.K_w):
                    selected = (selected - 1) % len(SERVERS)
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    selected = (selected + 1) % len(SERVERS)
                elif event.key == pygame.K_RETURN:
                    return SERVERS[selected][1], sound_enabled
                elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return None, sound_enabled
            elif event.type == pygame.MOUSEMOTION:
                for i, rect in enumerate(option_rects):
                    if rect.collidepoint(event.pos):
                        selected = i
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if checkbox_rect.collidepoint(event.pos):
                    sound_enabled = not sound_enabled
                else:
                    for i, rect in enumerate(option_rects):
                        if rect.collidepoint(event.pos):
                            return SERVERS[i][1], sound_enabled

        screen.fill(BG)
        for x in range(0, sw + 100, 100):
            pygame.draw.line(screen, GRID, (x, 0), (x, sh))
        for y in range(0, sh + 100, 100):
            pygame.draw.line(screen, GRID, (0, y), (sw, y))

        title = font_xl.render("ML-GAME", True, (220, 220, 220))
        screen.blit(title, (cx - title.get_width() // 2, sh // 3 - 80))

        sep_y = sh // 3 + 10
        pygame.draw.line(screen, DARK, (cx - 240, sep_y), (cx + 240, sep_y), 1)
        sub = font_sm.render("Server auswaehlen", True, GRAY)
        screen.blit(sub, (cx - sub.get_width() // 2, sep_y + 12))

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
                tx, ty = cx - 238, entry_y + 10
                pygame.draw.polygon(screen, WHITE,
                                    [(tx, ty), (tx + 8, ty + 7), (tx, ty + 14)])

            col_l = WHITE if is_sel else GRAY
            col_u = GRAY  if is_sel else DIMGRAY
            screen.blit(font_lg.render(label, True, col_l), (cx - 220, entry_y))
            screen.blit(font_sm.render(url,   True, col_u),
                        (cx - 220, entry_y + font_lg.get_height() + 2))

        # Sound-Checkbox
        cb_x, cb_y = 24, sh - 44
        checkbox_rect = pygame.Rect(cb_x, cb_y, 14, 14)
        pygame.draw.rect(screen, GRAY, checkbox_rect, 1)
        if sound_enabled:
            pygame.draw.line(screen, GRAY, (cb_x + 2, cb_y + 7),  (cb_x + 5,  cb_y + 11), 2)
            pygame.draw.line(screen, GRAY, (cb_x + 5, cb_y + 11), (cb_x + 12, cb_y + 3),  2)
        screen.blit(font_sm.render("Sound", True, GRAY), (cb_x + 20, cb_y))

        hint = font_sm.render(
            "hoch/runter  Auswaehlen      Enter  Weiter      ESC  Beenden",
            True, (45, 45, 45),
        )
        screen.blit(hint, (cx - hint.get_width() // 2, sh - 44))
        pygame.display.flip()


# ── Menü: Room-Auswahl ────────────────────────────────────────────────────────

def run_room_menu(screen: pygame.Surface, clock: pygame.time.Clock,
                  server_url: str,
                  font_sm: pygame.font.Font, font_lg: pygame.font.Font,
                  font_xl: pygame.font.Font) -> Optional[tuple[str, bool]]:
    """Gibt (room_name, spectator) zurück oder None für zurück zur Server-Auswahl."""

    fetch_result: dict = {"done": False, "data": None, "error": None}
    threading.Thread(
        target=_fetch_rooms_async, args=(server_url, fetch_result), daemon=True,
    ).start()

    selected    = 0
    spectator   = False
    custom_text = ""
    in_custom   = False
    row_rects:  list[pygame.Rect] = []

    while True:
        clock.tick(60)
        sw, sh = screen.get_size()
        cx = sw // 2

        rooms_list: list[tuple[str, int]] = []
        if fetch_result["data"]:
            for rid, info in fetch_result["data"].items():
                rooms_list.append((rid, info.get("players", 0)))

        total_rows = len(rooms_list) + 1   # +1 für Eigener-Room-Zeile
        selected   = max(0, min(selected, total_rows - 1))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if in_custom:
                        in_custom = False
                    else:
                        return None

                elif event.key == pygame.K_TAB:
                    spectator = not spectator

                elif event.key in (pygame.K_UP, pygame.K_w) and not in_custom:
                    selected = (selected - 1) % total_rows

                elif event.key in (pygame.K_DOWN, pygame.K_s) and not in_custom:
                    selected = (selected + 1) % total_rows

                elif event.key == pygame.K_RETURN:
                    if selected < len(rooms_list):
                        return (rooms_list[selected][0], spectator)
                    else:
                        return (custom_text.strip() or "default", spectator)

                elif event.key == pygame.K_BACKSPACE and in_custom:
                    custom_text = custom_text[:-1]

                elif (in_custom or selected == len(rooms_list)):
                    if event.unicode.isprintable() and len(custom_text) < 28:
                        custom_text += event.unicode
                        in_custom = True

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, rect in enumerate(row_rects):
                    if rect.collidepoint(event.pos):
                        if i == selected:
                            if i < len(rooms_list):
                                return (rooms_list[i][0], spectator)
                            else:
                                in_custom = True
                        else:
                            selected  = i
                            in_custom = (i == len(rooms_list))

            elif event.type == pygame.MOUSEMOTION and not in_custom:
                for i, rect in enumerate(row_rects):
                    if rect.collidepoint(event.pos):
                        selected = i

        # ── Zeichnen ───────────────────────────────────────────────────────────
        screen.fill(BG)
        for x in range(0, sw + 100, 100):
            pygame.draw.line(screen, GRID, (x, 0), (x, sh))
        for y in range(0, sh + 100, 100):
            pygame.draw.line(screen, GRID, (0, y), (sw, y))

        title = font_xl.render("ML-GAME", True, (220, 220, 220))
        screen.blit(title, (cx - title.get_width() // 2, sh // 4 - 60))

        srv_lbl = font_sm.render(server_url, True, DIMGRAY)
        screen.blit(srv_lbl, (cx - srv_lbl.get_width() // 2, sh // 4 + 10))

        sep_y = sh // 4 + 36
        pygame.draw.line(screen, DARK, (cx - 240, sep_y), (cx + 240, sep_y), 1)
        sub = font_sm.render("Room auswaehlen", True, GRAY)
        screen.blit(sub, (cx - sub.get_width() // 2, sep_y + 12))

        ROW_H  = 52
        LIST_Y = sep_y + 44

        if not fetch_result["done"]:
            loading = font_sm.render("Lade Rooms...", True, DIMGRAY)
            screen.blit(loading, (cx - loading.get_width() // 2, LIST_Y + 10))
        elif fetch_result["error"] and not rooms_list:
            err = font_sm.render("Server nicht erreichbar", True, (70, 45, 45))
            screen.blit(err, (cx - err.get_width() // 2, LIST_Y + 10))

        row_rects.clear()

        for i, (room_name, player_count) in enumerate(rooms_list):
            is_sel = (i == selected)
            ry     = LIST_Y + i * ROW_H
            rect   = pygame.Rect(cx - 250, ry - 6, 500, ROW_H - 4)
            row_rects.append(rect)

            if is_sel:
                hl = pygame.Surface((500, ROW_H - 4), pygame.SRCALPHA)
                hl.fill((255, 255, 255, 10))
                screen.blit(hl, (cx - 250, ry - 6))
                pygame.draw.rect(screen, DARK, rect, 1)
                tx, ty = cx - 238, ry + 8
                pygame.draw.polygon(screen, WHITE,
                                    [(tx, ty), (tx + 8, ty + 7), (tx, ty + 14)])

            col_name  = WHITE if is_sel else GRAY
            col_count = GRAY  if is_sel else DIMGRAY
            screen.blit(font_lg.render(room_name, True, col_name), (cx - 218, ry))
            cnt_lbl = font_sm.render(f"{player_count} Spieler", True, col_count)
            screen.blit(cnt_lbl, (cx + 180 - cnt_lbl.get_width(), ry + 6))

        # Eigener-Room-Zeile
        custom_idx = len(rooms_list)
        is_csel    = (selected == custom_idx)
        cry        = LIST_Y + custom_idx * ROW_H
        crect      = pygame.Rect(cx - 250, cry - 6, 500, ROW_H - 4)
        row_rects.append(crect)

        if is_csel:
            hl = pygame.Surface((500, ROW_H - 4), pygame.SRCALPHA)
            hl.fill((255, 255, 255, 8))
            screen.blit(hl, (cx - 250, cry - 6))
            pygame.draw.rect(screen, DARK, crect, 1)
            tx, ty = cx - 238, cry + 8
            pygame.draw.polygon(screen, WHITE,
                                [(tx, ty), (tx + 8, ty + 7), (tx, ty + 14)])

        blink       = in_custom and (pygame.time.get_ticks() // 500) % 2 == 0
        cursor_char = "|" if blink else ""
        custom_col  = WHITE if is_csel else GRAY
        custom_disp = font_lg.render(
            f"Eigener Room: {custom_text}{cursor_char}", True, custom_col)
        screen.blit(custom_disp, (cx - 218, cry))

        # Modus-Toggle
        mode_y = LIST_Y + total_rows * ROW_H + 14
        pygame.draw.line(screen, DARK, (cx - 240, mode_y - 6),
                         (cx + 240, mode_y - 6), 1)

        pl_col   = WHITE if not spectator else DIMGRAY
        sp_col   = WHITE if spectator     else DIMGRAY
        play_lbl = font_lg.render("SPIELEN",   True, pl_col)
        spec_lbl = font_lg.render("ZUSCHAUEN", True, sp_col)
        gap      = 28
        total_w  = play_lbl.get_width() + gap + spec_lbl.get_width()
        bx       = cx - total_w // 2
        screen.blit(play_lbl, (bx, mode_y + 8))
        pygame.draw.line(screen, DARK,
                         (bx + play_lbl.get_width() + gap // 2, mode_y + 12),
                         (bx + play_lbl.get_width() + gap // 2, mode_y + 36), 1)
        screen.blit(spec_lbl, (bx + play_lbl.get_width() + gap, mode_y + 8))

        tab_hint = font_sm.render("Tab: wechseln", True, (45, 45, 45))
        screen.blit(tab_hint, (cx - tab_hint.get_width() // 2, mode_y + 44))

        hint = font_sm.render(
            "hoch/runter  Auswaehlen  |  Tab  Modus  |  Enter  Verbinden  |  ESC  Zurueck",
            True, (45, 45, 45),
        )
        screen.blit(hint, (cx - hint.get_width() // 2, sh - 44))
        pygame.display.flip()


# ── Eingabe ───────────────────────────────────────────────────────────────────

def _aim_angle(screen: pygame.Surface) -> float:
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
             server_url: str, room: str, spectator: bool,
             sounds: SoundManager,
             font_sm: pygame.font.Font, font_md: pygame.font.Font,
             font_lg: pygame.font.Font, font_xl: pygame.font.Font) -> bool:
    """Führt die Spielschleife aus. Gibt True zurück wenn Rückkehr ins Menü."""
    global _my_id, _status

    _reset_state()

    qs = f"?room={room}"
    if spectator:
        qs += "&type=spectator"
    start_network(server_url + qs)

    caption = f"ml-game  –  {server_url}  [{room}]"
    if spectator:
        caption += "  [Zuschauer]"
    pygame.display.set_caption(caption)
    pygame.mouse.set_visible(spectator)
    sounds.start_music()

    cam_x, cam_y    = 0.0, 0.0
    effects:         list[dict] = []
    send_timer:      float = 0.0
    shoot_cooldown:  float = 0.0
    follow_mode:     bool  = True   # spectator: True = follow player, False = free WASD
    follow_idx:      int   = 0      # index into alive ships list

    try:
        while True:
            dt = clock.tick(FPS) / 1000.0
            shoot_cooldown = max(0.0, shoot_cooldown - dt)

            # ── Nachrichten vom WS-Thread ──────────────────────────────────────
            while not state_q.empty():
                msg      = state_q.get_nowait()
                msg_type = msg.get("type")

                if msg_type == "_connected":
                    base = f"Verbunden  |  {server_url}  [{room}]"
                    _status = ("Zuschauer  |  " + base) if spectator else base
                elif msg_type == "_disconnected":
                    _status = "Getrennt – verbinde neu..."
                    _my_id  = None
                elif msg_type == "welcome":
                    if not spectator:
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
                        if ev_type == "kill":
                            sounds.explosion()

            # ── Events ────────────────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        return True
                    elif spectator and event.key in (pygame.K_TAB, pygame.K_RIGHT):
                        alive = [s for s in _state.get("ships", []) if s.get("alive")]
                        if alive:
                            follow_idx  = (follow_idx + 1) % len(alive)
                            follow_mode = True
                    elif spectator and event.key == pygame.K_LEFT:
                        alive = [s for s in _state.get("ships", []) if s.get("alive")]
                        if alive:
                            follow_idx  = (follow_idx - 1) % len(alive)
                            follow_mode = True

            pressed = pygame.key.get_pressed()

            # ── Kamera ────────────────────────────────────────────────────────
            ships = _state.get("ships", [])
            if spectator:
                wasd = (pressed[pygame.K_w] or pressed[pygame.K_s] or
                        pressed[pygame.K_a] or pressed[pygame.K_d])
                if wasd:
                    follow_mode = False

                alive_ships = [s for s in ships if s.get("alive")]

                if follow_mode and alive_ships:
                    follow_idx = follow_idx % len(alive_ships)
                    followed   = alive_ships[follow_idx]
                    cam_x, cam_y = followed["x"], followed["y"]
                else:
                    if pressed[pygame.K_w]: cam_y -= SPECTATOR_CAM_SPEED * dt
                    if pressed[pygame.K_s]: cam_y += SPECTATOR_CAM_SPEED * dt
                    if pressed[pygame.K_a]: cam_x -= SPECTATOR_CAM_SPEED * dt
                    if pressed[pygame.K_d]: cam_x += SPECTATOR_CAM_SPEED * dt
            else:
                my_ship = next((s for s in ships if s["id"] == _my_id), None)
                if my_ship:
                    cam_x, cam_y = my_ship["x"], my_ship["y"]

            # ── Eingabe senden (30 Hz, nur als Spieler) + Sound ───────────────
            if not spectator:
                send_timer += dt
                if send_timer >= 1.0 / 30 and _my_id:
                    send_timer = 0.0
                    inp = _build_input(pressed, _aim_angle(screen))
                    input_q.put(inp)

                    if inp["shoot"] and shoot_cooldown == 0.0:
                        sounds.shoot()
                        shoot_cooldown = SHOOT_COOLDOWN

                    sounds.update_thrust(
                        inp["up"] or inp["down"] or inp["left"] or inp["right"]
                    )

            # ── Zeichnen ───────────────────────────────────────────────────────
            screen.fill(BG)
            draw_background(screen, cam_x, cam_y)
            draw_nebula(screen, cam_x, cam_y)
            draw_grid(screen, cam_x, cam_y)

            for b in _state.get("bullets", []):
                draw_bullet(screen, b, cam_x, cam_y)
            for s in ships:
                draw_ship(screen, s, s["id"] == _my_id, cam_x, cam_y)

            effects = [ef for ef in effects if draw_effect(screen, ef, dt, cam_x, cam_y)]

            if not spectator:
                draw_enemy_indicators(screen, ships, _my_id)
                my_ship = next((s for s in ships if s["id"] == _my_id), None)
                if my_ship and not my_ship["alive"]:
                    draw_death_overlay(screen, font_xl, font_md)
            else:
                sw, sh = screen.get_size()
                alive_ships = [s for s in ships if s.get("alive")]
                if follow_mode and alive_ships:
                    fi = follow_idx % len(alive_ships)
                    pid = alive_ships[fi]["id"][:8]
                    banner_txt = f"ZUSCHAUER  |  Spieler {fi + 1}/{len(alive_ships)} ({pid}…)  |  Tab/←→: wechseln  |  WASD: frei"
                elif follow_mode:
                    banner_txt = "ZUSCHAUER  |  Kein Spieler  |  Tab/←→: folgen  |  WASD: Kamera"
                else:
                    banner_txt = "ZUSCHAUER  |  WASD: Kamera  |  Tab/←→: Spieler folgen"
                banner = font_sm.render(banner_txt, True, (50, 50, 50))
                screen.blit(banner, (sw // 2 - banner.get_width() // 2, 16))

            draw_minimap(screen, ships, _my_id)
            draw_hud(screen, ships, _my_id, _status, font_sm, font_lg)

            if not spectator:
                mx, my_pos = pygame.mouse.get_pos()
                draw_crosshair(screen, mx, my_pos)

            pygame.display.flip()
    finally:
        sounds.stop_all()
        sounds.stop_music()
        stop_network()


# ── Einstiegspunkt ────────────────────────────────────────────────────────────

def main() -> None:
    pygame.init()
    pygame.mixer.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
    pygame.display.set_caption("ml-game")
    clock = pygame.time.Clock()

    font_sm = pygame.font.Font(None, 18)
    font_md = pygame.font.Font(None, 22)
    font_lg = pygame.font.Font(None, 30)
    font_xl = pygame.font.Font(None, 78)

    sounds = SoundManager()

    # CLI: Menü überspringen
    if len(sys.argv) > 1:
        srv  = sys.argv[1]
        room = sys.argv[2] if len(sys.argv) > 2 else "default"
        spec = len(sys.argv) > 3 and sys.argv[3] == "spectate"
        run_game(screen, clock, srv, room, spec, sounds,
                 font_sm, font_md, font_lg, font_xl)
        pygame.quit()
        return

    # Menü-Schleife: Server → Room → Spiel
    sound_enabled = True
    while True:
        pygame.mouse.set_visible(True)

        server_url, sound_enabled = run_server_menu(
            screen, clock, font_sm, font_lg, font_xl, sound_enabled)
        sounds.enabled = sound_enabled
        if server_url is None:
            break

        result = run_room_menu(screen, clock, server_url, font_sm, font_lg, font_xl)
        if result is None:
            continue   # zurück zur Server-Auswahl

        room, spectator = result
        back_to_menu = run_game(
            screen, clock, server_url, room, spectator, sounds,
            font_sm, font_md, font_lg, font_xl,
        )
        if not back_to_menu:
            break

    pygame.quit()


if __name__ == "__main__":
    main()
