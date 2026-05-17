"""Trainiertes Modell als Bot auf dem Server starten.

Zum Zuschauen: Client öffnen, selben Server + Room wählen, Modus "Zuschauen".

Usage:
  python run_bot.py
  python run_bot.py --model checkpoints/latest --room demo --server ws://localhost:3001/ws
  python run_bot.py --model checkpoints/latest --room demo --server ws://game.foerst.haus/ws
"""

import argparse
import json
import sys

from env import build_input, parse_obs
from stable_baselines3 import PPO
from websockets.sync.client import connect as ws_connect


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="checkpoints/latest", help="Pfad zum SB3-Modell"
    )
    parser.add_argument("--room", default="demo", help="Room-Name")
    parser.add_argument("--server", default="ws://localhost:3001/ws", help="Server-URL")
    args = parser.parse_args()

    print(f"Lade Modell:  {args.model}")
    model = PPO.load(args.model, device="cpu")

    url = f"{args.server}?type=bot&room={args.room}"
    print(f"Verbinde:     {url}")
    print(
        f"Zum Zuschauen Client öffnen → Server '{args.server}' → Room '{args.room}' → Zuschauen\n"
    )

    player_id = None
    kills = deaths = 0

    try:
        with ws_connect(url, open_timeout=15) as ws:
            while True:
                try:
                    raw = ws.recv(timeout=10.0)
                except TimeoutError:
                    continue

                msg = json.loads(raw)
                kind = msg.get("type")

                if kind == "welcome":
                    player_id = msg["player_id"]
                    print(f"Verbunden als {player_id[:8]}…")

                elif kind == "events":
                    for ev in msg.get("events", []):
                        if ev["type"] == "kill":
                            if ev.get("killer") == player_id:
                                kills += 1
                                print(f"  Kill!  K={kills}  D={deaths}")
                            if ev.get("victim") == player_id:
                                deaths += 1
                                print(f"  Tod    K={kills}  D={deaths}")

                elif kind == "observation":
                    obs = parse_obs(msg)
                    action = model.predict(obs, deterministic=True)[0]
                    try:
                        ws.send(json.dumps(build_input(action)))
                    except Exception:
                        print("Verbindung verloren.")
                        break

    except KeyboardInterrupt:
        pass

    print(f"\nErgebnis: {kills} Kills, {deaths} Tode")


if __name__ == "__main__":
    main()
