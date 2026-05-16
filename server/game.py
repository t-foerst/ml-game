import math
import random
import uuid
from typing import Optional

TICK_RATE = 60
SHIP_SPEED = 220.0  # px/s (WASD, Weltkoordinaten)
BULLET_SPEED = 620.0  # px/s
BULLET_LIFETIME = 4.0  # seconds
SHIP_RADIUS = 16.0  # Kollisionsradius px
SHOOT_COOLDOWN = 0.5  # seconds
SHIP_MAX_HEALTH = 3
RESPAWN_DELAY = 3.0  # seconds
SPAWN_RANGE = 500.0  # ±px Spawn-Bereich (klein = Bots starten nah beieinander)


class Bullet:
    def __init__(self, owner_id: str, x: float, y: float, angle: float):
        self.id = str(uuid.uuid4())
        self.owner_id = owner_id
        self.x = x
        self.y = y
        self.angle = angle
        self.age = 0.0

    def update(self, dt: float) -> None:
        self.x += math.cos(self.angle) * BULLET_SPEED * dt
        self.y += math.sin(self.angle) * BULLET_SPEED * dt
        self.age += dt

    @property
    def alive(self) -> bool:
        return self.age < BULLET_LIFETIME

    def serialize(self) -> dict:
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "x": self.x,
            "y": self.y,
            "angle": self.angle,
        }


class Ship:
    def __init__(self, ship_id: str):
        self.id = ship_id
        self.score = 0
        self.input: dict = {
            "up": False,
            "down": False,
            "left": False,
            "right": False,
            "shoot": False,
            "aim_angle": 0.0,
        }
        self._respawn()

    def _respawn(self) -> None:
        self.x = (random.random() - 0.5) * SPAWN_RANGE * 2
        self.y = (random.random() - 0.5) * SPAWN_RANGE * 2
        self.angle = random.uniform(0.0, math.pi * 2)  # Startrichtung zufällig
        self.health = SHIP_MAX_HEALTH
        self.alive = True
        self.shoot_cooldown = 0.0

    def apply_input(self, inp: dict) -> None:
        self.input = {
            "up": bool(inp.get("up")),
            "down": bool(inp.get("down")),
            "left": bool(inp.get("left")),
            "right": bool(inp.get("right")),
            "shoot": bool(inp.get("shoot")),
            "aim_angle": float(inp.get("aim_angle", self.angle)),
        }

    def update(self, dt: float) -> Optional["Bullet"]:
        if not self.alive:
            return None

        # Ausrichtung = Zielwinkel (unabhängig von der Bewegung)
        self.angle = self.input["aim_angle"]

        # Freie WASD-Bewegung in Weltkoordinaten
        vx, vy = 0.0, 0.0
        if self.input["up"]:
            vy -= SHIP_SPEED
        if self.input["down"]:
            vy += SHIP_SPEED
        if self.input["left"]:
            vx -= SHIP_SPEED
        if self.input["right"]:
            vx += SHIP_SPEED
        # Diagonalbewegung normalisieren (konstante Geschwindigkeit)
        if vx != 0.0 and vy != 0.0:
            vx *= 0.7071067811865476
            vy *= 0.7071067811865476
        self.x += vx * dt
        self.y += vy * dt

        self.shoot_cooldown = max(0.0, self.shoot_cooldown - dt)
        if self.input["shoot"] and self.shoot_cooldown == 0.0:
            self.shoot_cooldown = SHOOT_COOLDOWN
            offset = SHIP_RADIUS + 8.0
            return Bullet(
                self.id,
                self.x + math.cos(self.angle) * offset,
                self.y + math.sin(self.angle) * offset,
                self.angle,
            )
        return None

    def take_damage(self) -> None:
        if not self.alive:
            return
        self.health -= 1
        if self.health <= 0:
            self.alive = False

    def serialize(self) -> dict:
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "angle": self.angle,
            "health": self.health,
            "alive": self.alive,
            "score": self.score,
        }

    def get_observation(self, all_ships: list, bullets: list) -> dict:
        """ML-Schnittstelle: egozentrische Beobachtung (relative Koordinaten).

        Aktionsraum:
          up, down, left, right  – bool  – Bewegung in Weltkoordinaten
          shoot                  – bool  – Schießen
          aim_angle              – float – Zielwinkel in Radiant
        """
        enemies = [
            {
                "rel_x": s.x - self.x,
                "rel_y": s.y - self.y,
                "angle": s.angle,
                "health": s.health,
            }
            for s in all_ships
            if s.id != self.id and s.alive
        ]
        enemy_bullets = [
            {"rel_x": b.x - self.x, "rel_y": b.y - self.y, "angle": b.angle}
            for b in bullets
            if b.owner_id != self.id
        ]
        return {
            "self": {
                "x": self.x,
                "y": self.y,
                "angle": self.angle,
                "health": self.health,
                "shoot_cooldown": self.shoot_cooldown,
            },
            "enemies": enemies,
            "bullets": enemy_bullets,
        }


class Game:
    def __init__(self):
        self.ships: dict[str, Ship] = {}
        self.bullets: dict[str, Bullet] = {}
        self.tick = 0
        self._time = 0.0
        self._pending_respawns: list[tuple[float, str]] = []

    def add_player(self, player_id: str) -> Ship:
        ship = Ship(player_id)
        self.ships[player_id] = ship
        return ship

    def remove_player(self, player_id: str) -> None:
        self.ships.pop(player_id, None)

    def set_input(self, player_id: str, inp: dict) -> None:
        ship = self.ships.get(player_id)
        if ship:
            ship.apply_input(inp)

    def update(self, dt: float) -> list[dict]:
        self._time += dt
        self.tick += 1
        events: list[dict] = []

        # Respawns verarbeiten
        remaining = []
        for respawn_at, ship_id in self._pending_respawns:
            if self._time >= respawn_at and ship_id in self.ships:
                self.ships[ship_id]._respawn()
                events.append({"type": "respawn", "ship": ship_id})
            else:
                remaining.append((respawn_at, ship_id))
        self._pending_respawns = remaining

        # 1. Bestehende Geschosse bewegen + abgelaufene entfernen
        for b in list(self.bullets.values()):
            b.update(dt)
        for bid in [bid for bid, b in list(self.bullets.items()) if not b.alive]:
            del self.bullets[bid]

        # 2. Schiffe bewegen + neue Geschosse sammeln
        new_bullets: list[Bullet] = []
        for ship in list(self.ships.values()):
            bullet = ship.update(dt)
            if bullet:
                new_bullets.append(bullet)

        # 3. Kollision: bestehende Geschosse vs. Schiffe
        consumed: set[str] = set()
        for bullet in list(self.bullets.values()):
            if bullet.id in consumed:
                continue
            for ship in list(self.ships.values()):
                if not ship.alive or ship.id == bullet.owner_id:
                    continue
                dx = bullet.x - ship.x
                dy = bullet.y - ship.y
                if dx * dx + dy * dy < SHIP_RADIUS * SHIP_RADIUS:
                    consumed.add(bullet.id)
                    del self.bullets[bullet.id]
                    ship.take_damage()
                    if not ship.alive:
                        killer = self.ships.get(bullet.owner_id)
                        if killer:
                            killer.score += 1
                        events.append(
                            {
                                "type": "kill",
                                "killer": bullet.owner_id,
                                "victim": ship.id,
                            }
                        )
                        self._pending_respawns.append(
                            (self._time + RESPAWN_DELAY, ship.id)
                        )
                    else:
                        events.append(
                            {"type": "hit", "ship": ship.id, "shooter": bullet.owner_id}
                        )
                    break

        # 4. Neue Geschosse einfügen (werden erst nächsten Tick bewegt)
        for b in new_bullets:
            self.bullets[b.id] = b

        return events

    def get_state(self) -> dict:
        return {
            "tick": self.tick,
            "ships": [s.serialize() for s in self.ships.values()],
            "bullets": [b.serialize() for b in self.bullets.values()],
        }

    def get_observation(self, player_id: str) -> Optional[dict]:
        """ML-Schnittstelle: egozentrische Beobachtung für einen Agenten."""
        ship = self.ships.get(player_id)
        if not ship:
            return None
        return ship.get_observation(
            list(self.ships.values()),
            list(self.bullets.values()),
        )
