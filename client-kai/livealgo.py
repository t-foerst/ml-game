"""livealgo.py — AlgoAgent spielt live auf dem Server.

Verbindet sich als Bot und steuert ihn mit dem gewählten AlgoAgent.
So kann ein Mensch (normaler Client) gegen einen AlgoAgent kämpfen.

Start:
  python livealgo.py
"""

import asyncio
import json
import sys

import websockets

from algoagents import choose_agent, choose_server


async def run(server_url: str, algo_agent) -> None:
    bot_url = f"{server_url}?type=bot"
    print(f"[livealgo] Verbinde mit {bot_url} ...")

    while True:
        try:
            async with websockets.connect(bot_url) as ws:
                my_id: str | None = None
                print(f"[livealgo] Verbunden — Gegner: {algo_agent.name}")

                async for raw in ws:
                    msg   = json.loads(raw)
                    mtype = msg.get("type")

                    if mtype == "welcome":
                        my_id = msg["player_id"]
                        print(f"[livealgo] Bot-ID: {my_id}")
                        algo_agent.reset()

                    elif mtype == "observation":
                        inp = algo_agent.get_input(msg)
                        await ws.send(json.dumps({"type": "input", **inp}))

                    elif mtype == "events":
                        for ev in msg.get("events", []):
                            if ev.get("type") == "kill":
                                if ev.get("victim") == my_id:
                                    print("[livealgo] Abgeschossen!")
                                    algo_agent.reset()
                                elif ev.get("killer") == my_id:
                                    print("[livealgo] Kill!")

        except KeyboardInterrupt:
            print("\n[livealgo] Beendet.")
            return
        except Exception as e:
            print(f"[livealgo] Verbindungsfehler: {e} — verbinde neu...")
            await asyncio.sleep(2)


if __name__ == "__main__":
    algo_agent = choose_agent()
    url = sys.argv[1] if len(sys.argv) > 1 else choose_server()
    try:
        asyncio.run(run(url, algo_agent))
    except KeyboardInterrupt:
        print("\n[livealgo] Beendet.")
