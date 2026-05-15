import asyncio
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from game import Game, TICK_RATE

BROADCAST_RATE = 20  # Hz (State-Updates an Clients)

game = Game()
# player_id -> { ws: WebSocket, is_bot: bool }
players: dict[str, dict] = {}


async def _send(ws: WebSocket, data: dict) -> None:
    try:
        await ws.send_text(json.dumps(data))
    except Exception:
        pass


async def _broadcast(data: dict) -> None:
    """Sendet an alle verbundenen Clients; entfernt tote Verbindungen."""
    msg = json.dumps(data)
    dead: list[str] = []
    for pid, info in list(players.items()):
        try:
            await info["ws"].send_text(msg)
        except Exception:
            dead.append(pid)
    for pid in dead:
        players.pop(pid, None)
        game.remove_player(pid)


async def game_loop() -> None:
    interval = 1.0 / TICK_RATE
    broadcast_every = max(1, TICK_RATE // BROADCAST_RATE)  # alle N Ticks broadcasten
    loop = asyncio.get_running_loop()
    last_t = loop.time()

    while True:
        t0 = loop.time()
        dt = min(t0 - last_t, 0.05)  # max 50 ms (Schutz vor Sprüngen)
        last_t = t0

        events = game.update(dt)

        if game.tick % broadcast_every == 0:
            await _broadcast({"type": "state", **game.get_state()})

            # Für Bot-Clients: personalisierte Beobachtung senden
            # (Grundlage für späteres ML-Training)
            for pid, info in list(players.items()):
                if info["is_bot"]:
                    obs = game.get_observation(pid)
                    if obs:
                        await _send(info["ws"], {"type": "observation", **obs})

        if events:
            await _broadcast({"type": "events", "events": events})

        elapsed = loop.time() - t0
        await asyncio.sleep(max(0.0, interval - elapsed))


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(game_loop())
    yield


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    player_id = str(uuid.uuid4())

    # Bot-Clients verbinden mit ?type=bot
    is_bot = ws.query_params.get("type") == "bot"

    players[player_id] = {"ws": ws, "is_bot": is_bot}
    game.add_player(player_id)

    await _send(ws, {"type": "welcome", "player_id": player_id, "is_bot": is_bot})
    await _broadcast({"type": "join", "player_id": player_id})

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if msg.get("type") == "input":
                game.set_input(player_id, msg)
    except WebSocketDisconnect:
        pass
    finally:
        players.pop(player_id, None)
        game.remove_player(player_id)
        await _broadcast({"type": "leave", "player_id": player_id})
