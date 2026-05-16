import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from game import TICK_RATE, Game

BROADCAST_RATE = int(os.environ.get("BROADCAST_RATE", "60"))
SPEED_MULTIPLIER = float(os.environ.get("SPEED_MULTIPLIER", "1.0"))

DEFAULT_ROOM = "default"

# Fixiertes dt für deterministisches Training (kein Jitter durch Systemlast)
FIXED_DT = 1.0 / TICK_RATE


class Room:
    def __init__(self, room_id: str) -> None:
        self.room_id = room_id
        self.game = Game()
        self.players: dict[str, dict] = {}  # player_id → {ws, is_bot}
        self.spectators: dict[str, WebSocket] = {}  # spectator_id → ws
        self._task: Optional[asyncio.Task] = None
        self._bg_tasks: set[asyncio.Task] = set()  # pending fire-and-forget sends

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name=f"room-{self.room_id}")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def _fire(self, ws: WebSocket, msg: str) -> None:
        """Fire-and-forget send — never blocks the game loop."""
        async def _do() -> None:
            try:
                await ws.send_text(msg)
            except Exception:
                pass
        task = asyncio.create_task(_do())
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def send(self, ws: WebSocket, data: dict) -> None:
        self._fire(ws, json.dumps(data))

    async def broadcast(self, data: dict) -> None:
        msg = json.dumps(data)
        for pid, info in list(self.players.items()):
            self._fire(info["ws"], msg)
        for sid, ws in list(self.spectators.items()):
            self._fire(ws, msg)

    async def _loop(self) -> None:
        broadcast_every = max(1, TICK_RATE // BROADCAST_RATE)
        # SPEED_MULTIPLIER > 1 → schnellere Simulation (z. B. 10× für Training)
        sleep_dt = FIXED_DT / SPEED_MULTIPLIER
        loop = asyncio.get_running_loop()

        while True:
            t0 = loop.time()
            events = self.game.update(FIXED_DT)

            if self.game.tick % broadcast_every == 0:
                await self.broadcast({"type": "state", **self.game.get_state()})
                for pid, info in list(self.players.items()):
                    if info["is_bot"]:
                        obs = self.game.get_observation(pid)
                        if obs:
                            await self.send(info["ws"], {"type": "observation", **obs})

            if events:
                await self.broadcast({"type": "events", "events": events})

            elapsed = loop.time() - t0
            await asyncio.sleep(max(0.0, sleep_dt - elapsed))


# ── Room-Registry ─────────────────────────────────────────────────────────────

rooms: dict[str, Room] = {}


def get_or_create_room(room_id: str) -> Room:
    if room_id not in rooms:
        room = Room(room_id)
        room.start()
        rooms[room_id] = room
    return rooms[room_id]


async def _cleanup_loop() -> None:
    """Entfernt nicht-default Rooms, die seit 60 s keine Verbindungen mehr haben."""
    while True:
        await asyncio.sleep(60)
        to_remove = [
            rid
            for rid, r in list(rooms.items())
            if rid != DEFAULT_ROOM and not r.players and not r.spectators
        ]
        for rid in to_remove:
            await rooms.pop(rid).stop()


# ── App-Lifecycle ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_or_create_room(DEFAULT_ROOM)
    asyncio.create_task(_cleanup_loop(), name="room-cleanup")
    yield
    for room in list(rooms.values()):
        await room.stop()
    rooms.clear()


app = FastAPI(lifespan=lifespan)


# ── HTTP-Endpoints ────────────────────────────────────────────────────────────


@app.get("/rooms")
async def list_rooms() -> JSONResponse:
    return JSONResponse(
        {
            rid: {
                "players": len(r.players),
                "spectators": len(r.spectators),
                "tick": r.game.tick,
                "speed_multiplier": SPEED_MULTIPLIER,
            }
            for rid, r in rooms.items()
        }
    )


# ── WebSocket ─────────────────────────────────────────────────────────────────


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    client_id = str(uuid.uuid4())
    client_type = ws.query_params.get(
        "type", "player"
    )  # "player" | "bot" | "spectator"
    room_id = ws.query_params.get("room", DEFAULT_ROOM)

    room = get_or_create_room(room_id)

    if client_type == "spectator":
        room.spectators[client_id] = ws
        await room.send(
            ws,
            {
                "type": "welcome",
                "player_id": client_id,
                "is_spectator": True,
                "room": room_id,
            },
        )
        try:
            while True:
                await ws.receive_text()  # drain keepalives / ignore input
        except WebSocketDisconnect:
            pass
        finally:
            room.spectators.pop(client_id, None)
        return

    # ── Spieler / Bot ─────────────────────────────────────────────────────────
    is_bot = client_type == "bot"
    if is_bot:
        room.game.clear_bullets()
    room.players[client_id] = {"ws": ws, "is_bot": is_bot}
    room.game.add_player(client_id)

    await room.send(
        ws,
        {
            "type": "welcome",
            "player_id": client_id,
            "is_bot": is_bot,
            "room": room_id,
        },
    )
    await room.broadcast({"type": "join", "player_id": client_id})

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if msg.get("type") == "input":
                room.game.set_input(client_id, msg)
    except WebSocketDisconnect:
        pass
    finally:
        room.players.pop(client_id, None)
        room.game.remove_player(client_id)
        await room.broadcast({"type": "leave", "player_id": client_id})
