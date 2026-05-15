#!/usr/bin/env python3
"""client-kai — KI-Bot-Client für ml-game.

Verbindet sich als Bot (?type=bot) und empfängt egozentrische
Observations vom Server. Die KI-Logik gehört in decide().

Start:
  python main.py                          # Standardserver
  python main.py ws://localhost:3001/ws   # Lokaler Server
"""

import asyncio
import json
import sys

import websockets

SERVERS = [
    ("Lokal",       "ws://localhost:3001/ws"),
    ("Oeffentlich", "ws://game.foerst.haus/ws"),
]
DEFAULT_SERVER = SERVERS[1][1]


# ── KI-Logik ──────────────────────────────────────────────────────────────────

def decide(observation: dict) -> dict | None:
    """KI-Entscheidung auf Basis der Observation.

    Gibt einen Input-Dict zurück oder None (keine Aktion = steht still).

    observation enthält:
      self:    {x, y, angle, health, shoot_cooldown}
      enemies: [{rel_x, rel_y, angle, health}, ...]
      bullets: [{rel_x, rel_y, angle}, ...]
    """
    return None  # Platzhalter — Bot steht still


# ── Bot-Schleife ──────────────────────────────────────────────────────────────

async def run(server_url: str) -> None:
    bot_url = f"{server_url}?type=bot"
    print(f"Verbinde mit {bot_url} ...")

    async with websockets.connect(bot_url) as ws:
        print("Verbunden.")
        my_id: str | None = None

        async for raw in ws:
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "welcome":
                my_id = msg["player_id"]
                print(f"Bot-ID: {my_id}")

            elif msg_type == "observation":
                action = decide(msg)
                if action is not None:
                    await ws.send(json.dumps({"type": "input", **action}))

            elif msg_type == "events":
                for ev in msg.get("events", []):
                    if ev.get("type") == "kill" and ev.get("victim") == my_id:
                        print("Abgeschossen! Warte auf Respawn...")
                    elif ev.get("type") == "kill" and ev.get("killer") == my_id:
                        print("Kill!")


# ── Einstiegspunkt ────────────────────────────────────────────────────────────

def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SERVER
    try:
        asyncio.run(run(url))
    except KeyboardInterrupt:
        print("\nBot beendet.")


if __name__ == "__main__":
    main()
