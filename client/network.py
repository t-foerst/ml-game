"""WebSocket-Hintergrundthread.

Stellt zwei thread-sichere Queues bereit:
  state_q  – Nachrichten vom Server  (WS-Thread → Main-Thread)
  input_q  – Eingaben an den Server  (Main-Thread → WS-Thread)

stop_network()  – beendet aktive Verbindung und Reconnect-Loop sauber
start_network() – beendet vorherige Verbindung und startet neue Session
"""

import asyncio
import json
import queue
import threading
from typing import Callable, Optional

import websockets

state_q: queue.Queue = queue.Queue()
input_q: queue.Queue = queue.Queue()

# Wird vom laufenden WS-Thread gesetzt; thread-safe über call_soon_threadsafe
_stop_fn: Optional[Callable[[], None]] = None
_stop_fn_lock = threading.Lock()


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
    global _stop_fn
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    with _stop_fn_lock:
        _stop_fn = lambda: loop.call_soon_threadsafe(stop_event.set)

    while not stop_event.is_set():
        try:
            async with websockets.connect(url) as ws:
                state_q.put({"type": "_connected"})
                recv_task = asyncio.create_task(_recv(ws))
                send_task = asyncio.create_task(_send_loop(ws))
                stop_task = asyncio.create_task(stop_event.wait())

                _done, pending = await asyncio.wait(
                    [recv_task, send_task, stop_task],
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

        if not stop_event.is_set():
            state_q.put({"type": "_disconnected"})
            await asyncio.sleep(2.0)

    with _stop_fn_lock:
        _stop_fn = None


def stop_network() -> None:
    """Beendet Reconnect-Loop und aktive WS-Verbindung."""
    with _stop_fn_lock:
        fn = _stop_fn
    if fn:
        fn()


def start_network(url: str) -> None:
    """Beendet vorherige Verbindung und startet neue WS-Session."""
    stop_network()
    threading.Thread(
        target=lambda: asyncio.run(_ws_run(url)),
        daemon=True,
    ).start()
