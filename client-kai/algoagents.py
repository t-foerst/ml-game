"""algoagents.py — Algorithmus-gesteuerte Gegner (AlgoAgents).

Gemeinsame Basis für Training (agent.py / env.py) und Live-Spiel (livealgo.py).

Interface:
  obs = {"enemies": [{"rel_x", "rel_y", ...}], "bullets": [{"rel_x", "rel_y", "angle"}]}
  inp = agent.get_input(obs)   →  {"up", "down", "left", "right", "shoot", "aim_angle"}
"""

import math
import random

DODGE_RANGE = 150.0  # px — Kugeln innerhalb dieser Distanz ausweichen

SERVERS = [
    ("Lokal",       "ws://localhost:3001/ws"),
    ("Oeffentlich", "ws://game.foerst.haus/ws"),
]


# ── Basisklasse ───────────────────────────────────────────────────────────────

class AlgoAgent:
    name: str = "?"

    def reset(self) -> None:
        """Wird am Anfang jeder Episode aufgerufen."""

    def get_input(self, obs: dict) -> dict:
        """obs → {up, down, left, right, shoot, aim_angle}"""
        raise NotImplementedError


# ── Konkrete AlgoAgents ───────────────────────────────────────────────────────

class StillAgent(AlgoAgent):
    """Steht still, schießt nicht."""
    name = "Stillstehendes Ziel (kein Schuss)"

    def get_input(self, obs: dict) -> dict:
        return {"up": False, "down": False, "left": False, "right": False,
                "shoot": False, "aim_angle": 0.0}


class StillShooterAgent(AlgoAgent):
    """Steht still, schießt auf den nächsten Feind."""
    name = "Stillstehendes Ziel (schießt)"

    def get_input(self, obs: dict) -> dict:
        enemies = obs.get("enemies", [])
        aim_angle = 0.0
        if enemies:
            e = min(enemies, key=lambda e: e["rel_x"] ** 2 + e["rel_y"] ** 2)
            aim_angle = math.atan2(e["rel_y"], e["rel_x"])
        return {"up": False, "down": False, "left": False, "right": False,
                "shoot": bool(enemies), "aim_angle": aim_angle}


class LinearMoverAgent(AlgoAgent):
    """Bewegt sich horizontal hin und her, schießt nicht."""
    name = "Bewegtes Ziel, gerade Linie (kein Schuss)"

    def __init__(self, half_period_steps: int = 80):
        self._half_period = half_period_steps
        self._tick        = 0

    def reset(self) -> None:
        self._tick = 0

    def get_input(self, obs: dict) -> dict:
        self._tick += 1
        going_right = (self._tick // self._half_period) % 2 == 0
        return {"up": False, "down": False,
                "left": not going_right, "right": going_right,
                "shoot": False, "aim_angle": 0.0}


class OrbitDodgeAgent(AlgoAgent):
    """Kreist um den Agenten und weicht feindlichen Kugeln aus."""
    name = "Orbit + Dodge (Standardgegner)"

    def __init__(self):
        self._direction = 1  # +1 = CCW, -1 = CW

    def reset(self) -> None:
        self._direction = random.choice([-1, 1])

    def get_input(self, obs: dict) -> dict:
        enemies = obs.get("enemies", [])
        bullets = obs.get("bullets", [])

        aim_angle        = 0.0
        orbit_x, orbit_y = 1.0, 0.0

        if enemies:
            e  = min(enemies, key=lambda e: e["rel_x"] ** 2 + e["rel_y"] ** 2)
            dx, dy = e["rel_x"], e["rel_y"]
            dist = math.hypot(dx, dy)
            if dist > 1:
                aim_angle = math.atan2(dy, dx)
                orbit_x   = math.cos(aim_angle + self._direction * math.pi / 2)
                orbit_y   = math.sin(aim_angle + self._direction * math.pi / 2)

        dodge_x, dodge_y = 0.0, 0.0
        for b in bullets:
            bx, by = b["rel_x"], b["rel_y"]
            if bx ** 2 + by ** 2 > DODGE_RANGE ** 2:
                continue
            bdist = math.hypot(bx, by)
            if bdist < 1:
                continue
            bdir_x = math.cos(b["angle"])
            bdir_y = math.sin(b["angle"])
            if (bx * bdir_x + by * bdir_y) / bdist < -0.7:  # Kugel fliegt auf uns zu
                dodge_x -= bdir_y
                dodge_y += bdir_x

        mx = orbit_x + dodge_x
        my = orbit_y + dodge_y
        return {
            "up":        my < -0.3,
            "down":      my >  0.3,
            "left":      mx < -0.3,
            "right":     mx >  0.3,
            "shoot":     bool(enemies),
            "aim_angle": aim_angle,
        }


# ── Cycle-Agent ───────────────────────────────────────────────────────────────

class CycleAgent(AlgoAgent):
    """Durchläuft alle AlgoAgents der Reihe nach — bei jedem reset() nächster."""
    name = "Cycle (alle Agents nacheinander)"

    EPISODES_PER_AGENT = 20

    def __init__(self):
        self._agents  = [cls() for cls in SINGLE_AGENTS]
        self._index   = 0
        self._count   = 0
        self._current: AlgoAgent = self._agents[0]
        self._current.reset()

    def reset(self) -> None:
        self._count += 1
        if self._count > self.EPISODES_PER_AGENT:
            self._count  = 1
            self._index  = (self._index + 1) % len(self._agents)
            self._current = self._agents[self._index]
        self._current.reset()

    @property
    def active_name(self) -> str:
        return self._current.name

    def get_input(self, obs: dict) -> dict:
        return self._current.get_input(obs)


# ── Verfügbare Partner (Reihenfolge = Menü-Nummerierung) ─────────────────────

SINGLE_AGENTS: list[type[AlgoAgent]] = [
    StillAgent,
    StillShooterAgent,
    LinearMoverAgent,
    OrbitDodgeAgent,
]

ALGO_AGENTS: list[type[AlgoAgent]] = SINGLE_AGENTS + [CycleAgent]


# ── Auswahl-Helfer ────────────────────────────────────────────────────────────

def choose_agent() -> AlgoAgent:
    print("\nTrainingspartner / Gegner auswählen:")
    for i, cls in enumerate(ALGO_AGENTS):
        print(f"  [{i + 1}] {cls.name}")
    while True:
        try:
            idx = int(input("> ").strip()) - 1
            if 0 <= idx < len(ALGO_AGENTS):
                return ALGO_AGENTS[idx]()
        except (ValueError, EOFError):
            pass
        print(f"  Bitte 1–{len(ALGO_AGENTS)} eingeben.")


def choose_server() -> str:
    print("\nServer auswählen:")
    for i, (name, url) in enumerate(SERVERS):
        print(f"  [{i + 1}] {name}  ({url})")
    try:
        idx = int(input("> ").strip()) - 1
        if 0 <= idx < len(SERVERS):
            return SERVERS[idx][1]
    except (ValueError, EOFError):
        pass
    return SERVERS[0][1]
