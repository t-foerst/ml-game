"""WebSocket-Hintergrundthread.

Stellt zwei thread-sichere Queues bereit:
  state_q  – Nachrichten vom Server  (WS-Thread → Main-Thread)
  input_q  – Eingaben an den Server  (Main-Thread → WS-Thread)

Aufruf: start_network(url) startet den Thread einmalig.
"""

import asyncio
import json
import queue
import threading

import websockets

state_q: queue.Queue = queue.Queue()
input_q: queue.Queue = queue.Queue()


async def _recv(ws) -> None:
    async for raw in ws:
        state_q.put(json.loads(raw))


async def _send_loop(ws) -> None:
    while True:
        try:
            msg = input_q.get_nowait()
            await ws.send(json.dumps(msg))
        except queue.Empty:
            pass
        await asyncio.sleep(1.0 / 60)


async def _ws_run(url: str) -> None:
    while True:
        try:
            async with websockets.connect(url) as ws:
                state_q.put({"type": "_connected"})
                recv_task = asyncio.create_task(_recv(ws))
                send_task = asyncio.create_task(_send_loop(ws))
                _done, pending = await asyncio.wait(
                    [recv_task, send_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass
        except Exception:
            pass
        state_q.put({"type": "_disconnected"})
        await asyncio.sleep(2.0)


def start_network(url: str) -> None:
    """Startet den WebSocket-Thread. Pro Session einmal aufrufen."""
    threading.Thread(
        target=lambda: asyncio.run(_ws_run(url)),
        daemon=True,
    ).start()
