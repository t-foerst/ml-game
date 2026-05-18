"""Alle Zeichenfunktionen für den ml-game Client."""

import math
from pathlib import Path
from typing import Optional

import pygame
from constants import (
    BG,
    DARK,
    DIMGRAY,
    GRAY,
    GRID,
    HP_HIGH,
    HP_LOW,
    HP_MID,
    INDICATOR_DIST,
    MAX_HP,
    MINIMAP_SCALE,
    MINIMAP_SIZE,
    ORIGIN,
    WALL_COLOR,
    WHITE,
    WORLD_SIZE,
)

# ── Parallax-Hintergrund ──────────────────────────────────────────────────────

_ASSETS = Path(__file__).parent / "assets"
_bg_surf: Optional[pygame.Surface] = None
_nebula_surf: Optional[pygame.Surface] = None
_bg_scaled: Optional[pygame.Surface] = None
_bg_scaled_size: tuple[int, int] = (0, 0)

_BG_PARALLAX = 0.03
_NEBULA_PARALLAX = 0.12
_BG_PAD = 100


def _load_backgrounds() -> None:
    global _bg_surf, _nebula_surf
    if _bg_surf is None:
        try:
            _bg_surf = pygame.image.load(
                str(_ASSETS / "background-green.jpg")
            ).convert()
        except Exception:
            pass
    if _nebula_surf is None:
        try:
            _nebula_surf = pygame.image.load(
                str(_ASSETS / "nebula.png")
            ).convert_alpha()
        except Exception:
            pass


def _draw_tiled(
    surf: pygame.Surface, img: pygame.Surface, cam_x: float, cam_y: float, factor: float
) -> None:
    iw, ih = img.get_size()
    sw, sh = surf.get_size()
    ox = int(cam_x * factor) % iw
    oy = int(cam_y * factor) % ih
    x = -ox
    while x < sw:
        y = -oy
        while y < sh:
            surf.blit(img, (x, y), special_flags=pygame.BLEND_ADD)
            y += ih
        x += iw


def draw_background(surf: pygame.Surface, cam_x: float, cam_y: float) -> None:
    global _bg_scaled, _bg_scaled_size
    _load_backgrounds()
    if _bg_surf is None:
        return

    sw, sh = surf.get_size()
    target = (sw + _BG_PAD * 2, sh + _BG_PAD * 2)
    if target != _bg_scaled_size:
        _bg_scaled = pygame.transform.smoothscale(_bg_surf, target)
        _bg_scaled_size = target

    half = _BG_PAD
    ox = max(-half, min(half, int(cam_x * _BG_PARALLAX)))
    oy = max(-half, min(half, int(cam_y * _BG_PARALLAX)))
    surf.blit(_bg_scaled, (-half - ox, -half - oy))


def draw_nebula(surf: pygame.Surface, cam_x: float, cam_y: float) -> None:
    _load_backgrounds()
    if _nebula_surf:
        _draw_tiled(surf, _nebula_surf, cam_x, cam_y, _NEBULA_PARALLAX)


# ── Koordinaten-Hilfen ────────────────────────────────────────────────────────


def w2s(
    wx: float, wy: float, cam_x: float, cam_y: float, sw: int, sh: int
) -> tuple[int, int]:
    return (int(wx - cam_x + sw // 2), int(wy - cam_y + sh // 2))


def rotate_pts(
    pts: list[tuple[float, float]], angle: float, ox: float, oy: float
) -> list[tuple[float, float]]:
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return [
        (ox + px * cos_a - py * sin_a, oy + px * sin_a + py * cos_a) for px, py in pts
    ]


# ── Spielfeld ─────────────────────────────────────────────────────────────────


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

    ox, oy = w2s(0, 0, cam_x, cam_y, sw, sh)
    pygame.draw.line(surf, ORIGIN, (ox - 12, oy), (ox + 12, oy), 2)
    pygame.draw.line(surf, ORIGIN, (ox, oy - 12), (ox, oy + 12), 2)


def draw_boundary(surf: pygame.Surface, cam_x: float, cam_y: float) -> None:
    sw, sh = surf.get_size()
    x0, y0 = w2s(-WORLD_SIZE, -WORLD_SIZE, cam_x, cam_y, sw, sh)
    x1, y1 = w2s( WORLD_SIZE,  WORLD_SIZE, cam_x, cam_y, sw, sh)
    rect = pygame.Rect(x0, y0, x1 - x0, y1 - y0)
    r, g, b = WALL_COLOR
    for thickness, alpha in ((8, 25), (4, 60), (2, 140)):
        tmp = pygame.Surface((sw, sh), pygame.SRCALPHA)
        pygame.draw.rect(tmp, (r, g, b, alpha), rect, thickness)
        surf.blit(tmp, (0, 0))


# ── Spielerfarben ─────────────────────────────────────────────────────────────

_PLAYER_COLORS = [
    (220, 50, 50),
    (160, 60, 220),
    (230, 210, 40),
    (50, 120, 220),
    (50, 200, 80),
    (230, 130, 30),
    (40, 210, 210),
    (220, 60, 160),
]

_UFO_BODY = (40, 90, 200)
_UFO_RIM = (20, 50, 140)

_color_registry: dict[str, int] = {}


def _player_color(ship_id: str) -> tuple[int, int, int]:
    if ship_id not in _color_registry:
        _color_registry[ship_id] = len(_color_registry) % len(_PLAYER_COLORS)
    return _PLAYER_COLORS[_color_registry[ship_id]]


# ── Raumschiff ────────────────────────────────────────────────────────────────


def draw_ship(
    surf: pygame.Surface, ship: dict, is_me: bool, cam_x: float, cam_y: float
) -> None:
    if not ship["alive"]:
        return

    sw, sh = surf.get_size()
    sx, sy = w2s(ship["x"], ship["y"], cam_x, cam_y, sw, sh)
    angle = ship["angle"]

    # UFO-Körper (dreht sich nicht)
    pygame.draw.circle(surf, _UFO_BODY, (sx, sy), 20)
    pygame.draw.circle(surf, _UFO_RIM, (sx, sy), 20, 2)

    # Geschützturm (dreht sich mit aim_angle, Spielerfarbe)
    pcolor = _player_color(ship["id"])

    def rot(pts):
        return rotate_pts(pts, angle, sx, sy)

    body = rot([(16, 0), (-9, -10), (-9, 10)])
    cockpit = rot([(9, 0), (-3, -5), (-3, 5)])
    engine = rot([(-9, -4), (-14, -4), (-14, 4), (-9, 4)])

    pygame.draw.polygon(surf, pcolor, body)
    pygame.draw.polygon(surf, _UFO_RIM, body, 1)
    pygame.draw.polygon(surf, _UFO_RIM, cockpit)
    pygame.draw.polygon(surf, _UFO_RIM, engine)

    # Lebensbalken
    bw, bh = 36, 4
    bx, by = sx - bw // 2, sy - 36
    pygame.draw.rect(surf, (30, 30, 30), (bx, by, bw, bh))
    ratio = ship["health"] / MAX_HP
    bar_col = HP_HIGH if ratio > 0.6 else (HP_MID if ratio > 0.3 else HP_LOW)
    pygame.draw.rect(surf, bar_col, (bx, by, int(bw * ratio), bh))


# ── Feind-Richtungsindikatoren ────────────────────────────────────────────────


def draw_enemy_indicators(
    surf: pygame.Surface, ships: list, my_id: Optional[str]
) -> None:
    sw, sh = surf.get_size()
    cx, cy = sw // 2, sh // 2

    my_ship = next((s for s in ships if s["id"] == my_id), None)
    if not my_ship:
        return

    for ship in ships:
        if ship["id"] == my_id or not ship["alive"]:
            continue

        dx = ship["x"] - my_ship["x"]
        dy = ship["y"] - my_ship["y"]
        dist = math.hypot(dx, dy)
        if dist < 1:
            continue

        angle = math.atan2(dy, dx)
        ix = int(cx + math.cos(angle) * INDICATOR_DIST)
        iy = int(cy + math.sin(angle) * INDICATOR_DIST)

        brightness = max(60, min(180, int(200 - dist / 8)))
        col = (brightness, brightness, brightness)
        SIZE = 7
        pts = rotate_pts(
            [(SIZE, 0), (-SIZE * 0.7, -SIZE * 0.65), (-SIZE * 0.7, SIZE * 0.65)],
            angle,
            ix,
            iy,
        )
        pygame.draw.polygon(surf, col, pts)


# ── Geschoss ──────────────────────────────────────────────────────────────────


def draw_bullet(surf: pygame.Surface, b: dict, cam_x: float, cam_y: float) -> None:
    sw, sh = surf.get_size()
    sx, sy = w2s(b["x"], b["y"], cam_x, cam_y, sw, sh)

    for radius, alpha in [(14, 18), (9, 40), (6, 80), (4, 140)]:
        glow = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow, (255, 255, 255, alpha), (radius, radius), radius)
        surf.blit(glow, (sx - radius, sy - radius))

    pygame.draw.circle(surf, (255, 255, 255), (sx, sy), 2)


# ── Visuelle Effekte ──────────────────────────────────────────────────────────


def draw_effect(
    surf: pygame.Surface, ef: dict, dt: float, cam_x: float, cam_y: float
) -> bool:
    ef["age"] += dt
    if ef["age"] >= ef["duration"]:
        return False

    sw, sh = surf.get_size()
    sx, sy = w2s(ef["x"], ef["y"], cam_x, cam_y, sw, sh)
    p = ef["age"] / ef["duration"]
    alpha = int((1 - p) * 220)

    if ef["type"] == "kill":
        for radius, a_fac in [(int(65 * p), 1.0), (int(35 * p), 0.4)]:
            if radius < 1:
                continue
            tmp = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(
                tmp,
                (210, 210, 210, int(alpha * a_fac)),
                (radius + 2, radius + 2),
                radius,
                3,
            )
            surf.blit(tmp, (sx - radius - 2, sy - radius - 2))
    else:
        r = int(26 * p)
        if r > 0:
            tmp = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(tmp, (200, 200, 200, alpha), (r + 2, r + 2), r, 2)
            surf.blit(tmp, (sx - r - 2, sy - r - 2))

    return True


# ── Fadenkreuz ────────────────────────────────────────────────────────────────


def draw_crosshair(surf: pygame.Surface, mx: int, my: int) -> None:
    GAP  = 6    # Abstand vom Mittelpunkt bis Strich-Anfang
    LEN  = 18   # Strichlänge
    R    = 14   # Kreis-Radius
    W    = 2    # Strichbreite
    C    = (220, 220, 220)
    SH   = (0, 0, 0)

    # Schatten (1px versetzt) für Kontrast auf hellem Hintergrund
    pygame.draw.line(surf, SH, (mx - GAP - LEN + 1, my + 1), (mx - GAP + 1, my + 1), W)
    pygame.draw.line(surf, SH, (mx + GAP + 1,       my + 1), (mx + GAP + LEN + 1, my + 1), W)
    pygame.draw.line(surf, SH, (mx + 1, my - GAP - LEN + 1), (mx + 1, my - GAP + 1), W)
    pygame.draw.line(surf, SH, (mx + 1, my + GAP + 1),       (mx + 1, my + GAP + LEN + 1), W)
    pygame.draw.circle(surf, SH, (mx + 1, my + 1), R, 1)

    # Fadenkreuz
    pygame.draw.line(surf, C, (mx - GAP - LEN, my), (mx - GAP, my), W)
    pygame.draw.line(surf, C, (mx + GAP,       my), (mx + GAP + LEN, my), W)
    pygame.draw.line(surf, C, (mx, my - GAP - LEN), (mx, my - GAP), W)
    pygame.draw.line(surf, C, (mx, my + GAP),       (mx, my + GAP + LEN), W)
    pygame.draw.circle(surf, C, (mx, my), R, 1)
    pygame.draw.circle(surf, C, (mx, my), 2)


# ── Minimap ───────────────────────────────────────────────────────────────────


def draw_minimap(surf: pygame.Surface, ships: list, my_id: Optional[str]) -> None:
    S = MINIMAP_SIZE
    mm = pygame.Surface((S, S), pygame.SRCALPHA)
    mm.fill((0, 0, 0, 160))
    pygame.draw.rect(mm, (42, 42, 42, 255), mm.get_rect(), 1)

    my_ship = next((s for s in ships if s["id"] == my_id), None)
    cx = my_ship["x"] if my_ship else 0.0
    cy = my_ship["y"] if my_ship else 0.0

    for s in ships:
        if not s["alive"]:
            continue
        mx_ = int(S / 2 + (s["x"] - cx) * MINIMAP_SCALE)
        my_ = int(S / 2 + (s["y"] - cy) * MINIMAP_SCALE)
        if 0 <= mx_ < S and 0 <= my_ < S:
            col = _player_color(s["id"])
            r = 4 if s["id"] == my_id else 3
            pygame.draw.circle(mm, col, (mx_, my_), r)

    half = S // 2
    pygame.draw.line(mm, (40, 40, 40), (half - 6, half), (half + 6, half))
    pygame.draw.line(mm, (40, 40, 40), (half, half - 6), (half, half + 6))

    sw, sh = surf.get_size()
    surf.blit(mm, (sw - S - 16, sh - S - 16))


# ── HUD ───────────────────────────────────────────────────────────────────────


def draw_hud(
    surf: pygame.Surface,
    ships: list,
    my_id: Optional[str],
    status: str,
    font_sm: pygame.font.Font,
    font_lg: pygame.font.Font,
) -> None:
    surf_w = surf.get_width()

    surf.blit(font_sm.render(status, True, (68, 68, 68)), (16, 16))

    my_ship = next((s for s in ships if s["id"] == my_id), None)
    if my_ship:
        hp = "* " * my_ship["health"] + ". " * (MAX_HP - my_ship["health"])
        surf.blit(font_lg.render(hp.strip(), True, WHITE), (16, 36))

    hint = font_sm.render(
        "WASD: Bewegen  |  Maus: Zielen  |  LMB / Leertaste: Schiessen  |  ESC: Menue",
        True,
        (50, 50, 50),
    )
    surf.blit(hint, (16, 70))

    title = font_sm.render("PUNKTE", True, (48, 48, 48))
    surf.blit(title, (surf_w - title.get_width() - 16, 16))

    for i, s in enumerate(sorted(ships, key=lambda x: -x["score"])):
        you = "  <- du" if s["id"] == my_id else ""
        dead = "  x" if not s["alive"] else ""
        col = _player_color(s["id"])
        label = font_sm.render(f"{s['score']} kills{you}{dead}", True, col)
        surf.blit(label, (surf_w - label.get_width() - 16, 36 + i * 20))


# ── Tod-Overlay ───────────────────────────────────────────────────────────────


def draw_death_overlay(
    surf: pygame.Surface, font_xl: pygame.font.Font, font_md: pygame.font.Font
) -> None:
    overlay = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 140))
    surf.blit(overlay, (0, 0))

    sw, sh = surf.get_size()
    t1 = font_xl.render("ZERSTOERT", True, (190, 190, 190))
    t2 = font_md.render("Respawn in 3 Sekunden...", True, (75, 75, 75))
    surf.blit(t1, (sw // 2 - t1.get_width() // 2, sh // 2 - t1.get_height() // 2))
    surf.blit(t2, (sw // 2 - t2.get_width() // 2, sh // 2 + 44))
