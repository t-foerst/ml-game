"""Gymnasium-Environment für ml-game Training.

Wrapp server/game.py direkt — kein Netzwerk, kein Echtzeit-Limit.
Aktionsraum: [up, down, left, right, shoot, rotate(0=nix,1=links,2=rechts)]

Episode endet bei erstem Treffer (Agent oder Gegner) oder nach MAX_STEPS.
"""

import math
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).parent.parent / "server"))
from game import Game, SHOOT_COOLDOWN, SPAWN_RANGE, BULLET_SPEED
from algoagents import AlgoAgent, OrbitDodgeAgent

# Observation
N_ENEMIES = 1
OBS_SIZE  = N_ENEMIES * 4 + 2 + 3  # + nächste Kugel: rel_x, rel_y, angle

# Steuerung
ROTATE_SPEED = 0.15
STEP_DT      = 1 / 20

# Episode
MAX_STEPS = 300

# Rewards
R_HIT_LANDED =   10.0
R_HIT_TAKEN  =  -10.0
R_SURVIVE    =  0.01

# Annäherungs-/Orbit-Übergang
SIGHT_RANGE        = 400.0   # px, Grenze zwischen Annähern und Orbit
BLEND_WIDTH        = 300.0   # px, Übergangszone (±50 um SIGHT_RANGE)
R_APPROACH_SCALE   = 0.01   # Reward pro px Annäherung
R_TANGENTIAL_SCALE = 0.0     # Orbit deaktiviert — Agent soll direkt angreifen
R_AIM_PASSIVE      = 0.05    # Reward pro Schritt für richtigen Zielwinkel
R_AIM_SHOOT        = 2.0     # Zusatz-Reward wenn dabei auch geschossen wird
AIM_CONE_DEG       = 15.0    # Winkel-Toleranz in Grad (enger = präziser)
R_JERK             = 0.1     # Penalty bei Bewegungsrichtungsänderung


def obs_to_vec(raw: dict, prev_enemies: list | None = None,
               aim_angle: float = 0.0) -> np.ndarray:
    """[rel_x, rel_y, delta_x, delta_y, sin(aim_diff), cos(aim_diff)]"""
    vec = []

    enemies = sorted(raw["enemies"],
                     key=lambda e: e["rel_x"] ** 2 + e["rel_y"] ** 2)
    aim_diff = 0.0
    for i in range(N_ENEMIES):
        if i < len(enemies):
            e = enemies[i]
            rx, ry = e["rel_x"] / SPAWN_RANGE, e["rel_y"] / SPAWN_RANGE
            if prev_enemies and i < len(prev_enemies):
                p = prev_enemies[i]
                dx = (e["rel_x"] - p["rel_x"]) / SPAWN_RANGE
                dy = (e["rel_y"] - p["rel_y"]) / SPAWN_RANGE
            else:
                dx, dy = 0.0, 0.0
            vec += [rx, ry, dx, dy]
            if i == 0:
                aim_diff = aim_angle - math.atan2(e["rel_y"], e["rel_x"])
        else:
            vec += [0.0, 0.0, 0.0, 0.0]

    vec += [math.sin(aim_diff), math.cos(aim_diff)]

    bullets = raw.get("bullets", [])
    if bullets:
        b = min(bullets, key=lambda b: b["rel_x"] ** 2 + b["rel_y"] ** 2)
        vec += [b["rel_x"] / SPAWN_RANGE, b["rel_y"] / SPAWN_RANGE, b["angle"] / math.pi]
    else:
        vec += [0.0, 0.0, 0.0]

    return np.clip(np.array(vec, dtype=np.float32), -1.0, 1.0)


def action_to_input(action, aim_angle: float) -> tuple[dict, float]:
    up, down, left, right, shoot, rotate = action
    if rotate == 1:
        aim_angle -= ROTATE_SPEED
    elif rotate == 2:
        aim_angle += ROTATE_SPEED
    aim_angle %= 2 * math.pi
    return {
        "up": bool(up), "down": bool(down),
        "left": bool(left), "right": bool(right),
        "shoot": bool(shoot), "aim_angle": aim_angle,
    }, aim_angle


class MLGameEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, debug: bool = False, algo_agent: AlgoAgent | None = None):
        super().__init__()
        self.action_space      = spaces.MultiDiscrete([2, 2, 2, 2, 2, 3])
        self.observation_space = spaces.Box(-1., 1., (OBS_SIZE,), np.float32)
        self.debug             = debug
        self._algo_agent       = algo_agent if algo_agent is not None else OrbitDodgeAgent()

        self._game         = Game()
        self._agent_id     = "agent"
        self._opp_id       = "opponent"
        self._aim_angle    = 0.0
        self._prev_enemies   = None
        self._prev_agent_pos = None
        self._prev_opp_pos   = None
        self._prev_movement  = None
        self._step_count     = 0
        self._episode_num    = 0

        # Debug-Statistiken (pro Episode)
        self._ep_reward      = 0.0
        self._ep_hits_landed = 0
        self._ep_hits_taken  = 0
        self._ep_shots_fired = 0
        self._ep_dist_sum    = 0.0

        # Laufende Statistiken (über alle Episoden)
        self._total_episodes = 0
        self._wins           = 0   # Agent trifft zuerst
        self._losses         = 0   # Agent wird zuerst getroffen
        self._timeouts       = 0
        self._reward_history : list[float] = []

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._game = Game()
        self._game.add_player(self._agent_id)
        self._game.add_player(self._opp_id)
        self._aim_angle      = 0.0
        self._prev_enemies   = None
        self._prev_agent_pos = None
        self._prev_opp_pos   = None
        self._prev_movement  = None
        self._step_count     = 0
        self._episode_num   += 1
        self._algo_agent.reset()

        self._ep_reward      = 0.0
        self._ep_hits_landed = 0
        self._ep_hits_taken  = 0
        self._ep_shots_fired = 0
        self._ep_dist_sum    = 0.0

        raw = self._game.get_observation(self._agent_id)
        return obs_to_vec(raw, None, self._aim_angle), {}

    def step(self, action):
        self._step_count += 1
        up, down, left, right, shoot, rotate = action

        if shoot:
            self._ep_shots_fired += 1

        inp, self._aim_angle = action_to_input(action, self._aim_angle)
        movement = (up, down, left, right)
        if self._prev_movement is not None and movement != self._prev_movement:
            reward_jerk = -R_JERK
        else:
            reward_jerk = 0.0
        self._prev_movement = movement
        self._game.set_input(self._agent_id, inp)
        self._apply_opponent()
        events = self._game.update(STEP_DT)

        # Distanz zum Feind tracken
        agent  = self._game.ships.get(self._agent_id)
        opp    = self._game.ships.get(self._opp_id)
        reward = reward_jerk
        done   = False
        result = "timeout"

        if agent and opp:
            dx_eo = opp.x - agent.x
            dy_eo = opp.y - agent.y
            dist  = math.hypot(dx_eo, dy_eo)
            self._ep_dist_sum += dist

            if self._prev_agent_pos and self._prev_opp_pos and dist > 1:
                # Blend: t=1 weit (Annähern aktiv), t=0 nah (Orbit aktiv)
                t          = max(0.0, min(1.0, (dist - (SIGHT_RANGE - BLEND_WIDTH / 2)) / BLEND_WIDTH))
                blend_far  = t
                blend_near = 1.0 - t

                move_x   = agent.x - self._prev_agent_pos[0]
                move_y   = agent.y - self._prev_agent_pos[1]
                toward_x = dx_eo / dist
                toward_y = dy_eo / dist

                # 1) Annäherungs-Reward (bidirektional: Näher = +, Weiter = -)
                if blend_far > 0:
                    approach = move_x * toward_x + move_y * toward_y
                    reward += approach * R_APPROACH_SCALE * blend_far

                # 2) Tangential-Reward (Orbit)
                if blend_near > 0:
                    tangential = abs(-move_x * toward_y + move_y * toward_x)
                    reward += tangential * R_TANGENTIAL_SCALE * blend_near

                # 3) Ziel-Reward mit Vorhalt — nur im Nahbereich
                if blend_near > 0:
                    opp_vel_x = (opp.x - self._prev_opp_pos[0]) / STEP_DT
                    opp_vel_y = (opp.y - self._prev_opp_pos[1]) / STEP_DT
                    lead_time = dist / BULLET_SPEED
                    lead_dx   = dx_eo + opp_vel_x * lead_time
                    lead_dy   = dy_eo + opp_vel_y * lead_time
                    ideal     = math.atan2(lead_dy, lead_dx)
                    diff_rad  = abs(math.atan2(
                        math.sin(self._aim_angle - ideal),
                        math.cos(self._aim_angle - ideal),
                    ))
                    diff_deg  = math.degrees(diff_rad)
                    if diff_deg < AIM_CONE_DEG:
                        factor  = (1.0 - diff_deg / AIM_CONE_DEG) * blend_near
                        reward += factor * R_AIM_PASSIVE          # jeden Schritt
                        if shoot:
                            reward += factor * R_AIM_SHOOT        # Bonus beim Schuss

            self._prev_agent_pos = (agent.x, agent.y)
            if opp.alive:
                self._prev_opp_pos = (opp.x, opp.y)

        for ev in events:
            ev_type = ev.get("type")
            if ev_type == "hit":
                if ev["ship"] == self._agent_id:
                    reward += R_HIT_TAKEN
                    self._ep_hits_taken += 1
                    done   = True
                    result = "loss"
                elif ev["ship"] == self._opp_id:
                    reward += R_HIT_LANDED
                    self._ep_hits_landed += 1
                    done   = True
                    result = "win"
            elif ev_type == "kill":
                if ev["victim"] == self._agent_id:
                    reward += R_HIT_TAKEN
                    self._ep_hits_taken += 1
                    done   = True
                    result = "loss"
                elif ev["victim"] == self._opp_id:
                    reward += R_HIT_LANDED
                    self._ep_hits_landed += 1
                    done   = True
                    result = "win"

        # Timeout
        truncated = False
        if not done and self._step_count >= MAX_STEPS:
            truncated = True
            self._timeouts += 1
            result = "timeout"

        self._ep_reward += reward

        if done or truncated:
            self._total_episodes += 1
            if result == "win":
                self._wins += 1
            elif result == "loss":
                self._losses += 1
            self._reward_history.append(self._ep_reward)
            if len(self._reward_history) > 100:
                self._reward_history.pop(0)

            if self.debug:
                self._print_episode(result)

        raw = self._game.get_observation(self._agent_id)
        if raw:
            obs = obs_to_vec(raw, self._prev_enemies, self._aim_angle)
            self._prev_enemies = raw["enemies"]
        else:
            obs = np.zeros(OBS_SIZE, dtype=np.float32)

        return obs, reward, done, truncated, {}

    # ── Debug-Output ───────────────────────────────────────────────────────────

    def _print_episode(self, result: str) -> None:
        n            = self._total_episodes
        win_rate     = self._wins     / n * 100 if n > 0 else 0.0
        timeout_rate = self._timeouts / n * 100 if n > 0 else 0.0

        result_str  = {"win": "TREFFER ✓", "loss": "GETROFFEN ✗", "timeout": "TIMEOUT —"}[result]
        agent_name  = getattr(self._algo_agent, "active_name", self._algo_agent.name)

        print(
            f"Ep {n:>5} | {result_str:<12} | "
            f"Steps: {self._step_count:>3} | "
            f"Timeout: {timeout_rate:>5.1f}% | "
            f"Siege: {win_rate:>5.1f}% | "
            f"Gegner: {agent_name}"
        )

        if n % 100 == 0:
            print(f"\n{'='*60}")
            print(f"  Zusammenfassung nach {n} Episoden")
            print(f"  Siege: {self._wins}  |  Niederlagen: {self._losses}  |  Timeouts: {self._timeouts}")
            print(f"  Timeout: {timeout_rate:.1f}%  |  Siegrate: {win_rate:.1f}%")
            print(f"{'='*60}\n")


    # ── Gegner ────────────────────────────────────────────────────────────────

    def _apply_opponent(self) -> None:
        opp = self._game.ships.get(self._opp_id)
        if not opp or not opp.alive:
            return
        obs = self._game.get_observation(self._opp_id)
        if obs:
            self._game.set_input(self._opp_id, self._algo_agent.get_input(obs))
