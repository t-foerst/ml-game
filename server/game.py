import math
import uuid
import random
from typing import Optional

TICK_RATE = 60
TANK_SPEED = 200.0          # px/s vorwärts
TANK_REVERSE_SPEED = 100.0  # px/s rückwärts
TANK_ROTATION = 2.2         # rad/s
BULLET_SPEED = 600.0        # px/s
BULLET_LIFETIME = 4.0       # seconds
TANK_RADIUS = 18.0          # px (Kollisionsradius)
SHOOT_COOLDOWN = 0.5        # seconds
TANK_MAX_HEALTH = 3
RESPAWN_DELAY = 3.0         # seconds
SPAWN_RANGE = 1000.0        # ±px für Spawn-Bereich


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


class Tank:
    def __init__(self, tank_id: str):
        self.id = tank_id
        self.score = 0
        self.input: dict = {
            "forward": False,
            "backward": False,
            "left": False,
            "right": False,
            "shoot": False,
        }
        self._respawn()

    def _respawn(self) -> None:
        self.x = (random.random() - 0.5) * SPAWN_RANGE * 2
        self.y = (random.random() - 0.5) * SPAWN_RANGE * 2
        self.angle = random.uniform(0.0, math.pi * 2)
        self.health = TANK_MAX_HEALTH
        self.alive = True
        self.shoot_cooldown = 0.0

    def apply_input(self, inp: dict) -> None:
        self.input = {
            "forward":  bool(inp.get("forward")),
            "backward": bool(inp.get("backward")),
            "left":     bool(inp.get("left")),
            "right":    bool(inp.get("right")),
            "shoot":    bool(inp.get("shoot")),
        }

    def update(self, dt: float) -> Optional["Bullet"]:
        if not self.alive:
            return None

        if self.input["left"]:
            self.angle -= TANK_ROTATION * dt
        if self.input["right"]:
            self.angle += TANK_ROTATION * dt

        if self.input["forward"]:
            speed = TANK_SPEED
        elif self.input["backward"]:
            speed = -TANK_REVERSE_SPEED
        else:
            speed = 0.0

        self.x += math.cos(self.angle) * speed * dt
        self.y += math.sin(self.angle) * speed * dt

        self.shoot_cooldown = max(0.0, self.shoot_cooldown - dt)
        if self.input["shoot"] and self.shoot_cooldown == 0.0:
            self.shoot_cooldown = SHOOT_COOLDOWN
            offset = TANK_RADIUS + 8.0
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

    def get_observation(self, all_tanks: list, bullets: list) -> dict:
        """ML-Schnittstelle: egozentrische Beobachtung (relative Koordinaten).

        Aktionsraum: { forward, backward, left, right, shoot } (alle bool)
        Beobachtungsraum:
          self:    position, angle, health, shoot_cooldown
          enemies: Liste relative Positionen/Winkel
          bullets: Liste feindliche Geschosse (relativ)
        """
        enemies = [
            {
                "rel_x": t.x - self.x,
                "rel_y": t.y - self.y,
                "angle": t.angle,
                "health": t.health,
            }
            for t in all_tanks
            if t.id != self.id and t.alive
        ]
        enemy_bullets = [
            {
                "rel_x": b.x - self.x,
                "rel_y": b.y - self.y,
                "angle": b.angle,
            }
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
        self.tanks: dict[str, Tank] = {}
        self.bullets: dict[str, Bullet] = {}
        self.tick = 0
        self._time = 0.0
        self._pending_respawns: list[tuple[float, str]] = []

    def add_player(self, player_id: str) -> Tank:
        tank = Tank(player_id)
        self.tanks[player_id] = tank
        return tank

    def remove_player(self, player_id: str) -> None:
        self.tanks.pop(player_id, None)

    def set_input(self, player_id: str, inp: dict) -> None:
        tank = self.tanks.get(player_id)
        if tank:
            tank.apply_input(inp)

    def update(self, dt: float) -> list[dict]:
        self._time += dt
        self.tick += 1
        events: list[dict] = []

        # Respawns verarbeiten
        remaining = []
        for respawn_at, tank_id in self._pending_respawns:
            if self._time >= respawn_at and tank_id in self.tanks:
                self.tanks[tank_id]._respawn()
                events.append({"type": "respawn", "tank": tank_id})
            else:
                remaining.append((respawn_at, tank_id))
        self._pending_respawns = remaining

        # 1. Bestehende Geschosse bewegen + abgelaufene entfernen
        for bullet in list(self.bullets.values()):
            bullet.update(dt)
        for bid in [bid for bid, b in list(self.bullets.items()) if not b.alive]:
            del self.bullets[bid]

        # 2. Panzer bewegen, neue Geschosse sammeln
        new_bullets: list[Bullet] = []
        for tank in list(self.tanks.values()):
            bullet = tank.update(dt)
            if bullet:
                new_bullets.append(bullet)

        # 3. Kollisionserkennung: bestehende Geschosse vs. Panzer
        consumed: set[str] = set()
        for bullet in list(self.bullets.values()):
            if bullet.id in consumed:
                continue
            for tank in list(self.tanks.values()):
                if not tank.alive or tank.id == bullet.owner_id:
                    continue
                dx = bullet.x - tank.x
                dy = bullet.y - tank.y
                if dx * dx + dy * dy < TANK_RADIUS * TANK_RADIUS:
                    consumed.add(bullet.id)
                    del self.bullets[bullet.id]
                    tank.take_damage()
                    if not tank.alive:
                        killer = self.tanks.get(bullet.owner_id)
                        if killer:
                            killer.score += 1
                        events.append({"type": "kill", "killer": bullet.owner_id, "victim": tank.id})
                        self._pending_respawns.append((self._time + RESPAWN_DELAY, tank.id))
                    else:
                        events.append({"type": "hit", "tank": tank.id})
                    break

        # 4. Neu abgefeuerte Geschosse einfügen (werden erst nächsten Tick bewegt)
        for bullet in new_bullets:
            self.bullets[bullet.id] = bullet

        return events

    def get_state(self) -> dict:
        return {
            "tick": self.tick,
            "tanks": [t.serialize() for t in self.tanks.values()],
            "bullets": [b.serialize() for b in self.bullets.values()],
        }

    def get_observation(self, player_id: str) -> Optional[dict]:
        """ML-Schnittstelle: strukturierte egozentrische Beobachtung für einen Agenten."""
        tank = self.tanks.get(player_id)
        if not tank:
            return None
        return tank.get_observation(
            list(self.tanks.values()),
            list(self.bullets.values()),
        )
