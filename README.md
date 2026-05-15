# ml-game

Ein minimalistisches Multiplayer-Raumschiffspiel mit WebSocket-Server und ML-Schnittstelle.

- Schwarz-Weiß-Grafik, kein Asset-Loading
- Freie WASD-Bewegung unabhängig von der Zielrichtung
- Mausgesteuertes Zielen + Schießen
- Feind-Richtungsindikatoren um den Spieler
- ML-Bot-Interface über `?type=bot`

---

## Server starten

```yaml
# docker-compose.yml
# Download: https://raw.githubusercontent.com/t-foerst/ml-game/main/docker-compose.yml

services:
  server:
    image: ghcr.io/t-foerst/ml-game/server:latest
    ports:
      - "3001:3001"
    restart: unless-stopped
```

```bash
docker compose up -d
```

---

## Client

Vorgefertigte Binaries gibt es auf der [Releases-Seite](../../releases/latest):

| Plattform | Datei | Ausführen |
|-----------|-------|-----------|
| Linux     | `ml-game-linux.tar.gz` | `tar xzf ml-game-linux.tar.gz && ./ml-game` |
| Windows   | `ml-game-windows.exe` | Doppelklick |

Direkt aus dem Quellcode starten (erfordert Python 3.10+):

```bash
cd client
pip install -r requirements.txt
python main.py                          # Menü (Server-Auswahl)
python main.py ws://localhost:3001/ws   # Menü überspringen
```

---

## Steuerung

| Taste / Eingabe | Aktion |
|-----------------|--------|
| `W A S D` / Pfeiltasten | Bewegen (unabhängig von Zielrichtung) |
| Maus | Zielrichtung |
| Linke Maustaste / Leertaste | Schießen |
| `ESC` | Zurück ins Menü |

---

## ML-Bot-Interface

Ein Bot verbindet sich per WebSocket mit dem Query-Parameter `?type=bot`:

```
ws://localhost:3001/ws?type=bot
```

### Nachrichten vom Server

**`welcome`** – direkt nach Verbindungsaufbau:
```json
{ "type": "welcome", "player_id": "<uuid>", "is_bot": true }
```

**`state`** – Spielzustand, ~20 Hz:
```json
{
  "type": "state",
  "tick": 42,
  "ships": [
    { "id": "<uuid>", "x": 100.0, "y": -50.0, "angle": 1.57,
      "health": 3, "alive": true, "score": 2 }
  ],
  "bullets": [
    { "id": "<uuid>", "owner_id": "<uuid>", "x": 200.0, "y": 0.0, "angle": 0.0 }
  ]
}
```

**`observation`** – egozentrische Beobachtung, nur für Bots, ~20 Hz:
```json
{
  "type": "observation",
  "self": {
    "x": 0.0, "y": 0.0, "angle": 0.0,
    "health": 3, "shoot_cooldown": 0.0
  },
  "enemies": [
    { "rel_x": 300.0, "rel_y": -150.0, "angle": 3.14, "health": 2 }
  ],
  "bullets": [
    { "rel_x": 50.0, "rel_y": 10.0, "angle": 1.57 }
  ]
}
```

**`events`** – Treffer und Kills:
```json
{ "type": "events", "events": [
    { "type": "hit",  "ship": "<uuid>" },
    { "type": "kill", "killer": "<uuid>", "victim": "<uuid>" }
]}
```

### Nachrichten an den Server

**`input`** – Steuerbefehl (beliebig oft sendbar):
```json
{
  "type": "input",
  "up": true, "down": false, "left": false, "right": true,
  "shoot": false,
  "aim_angle": 1.047
}
```

`aim_angle` in Radiant (0 = rechts, π/2 = unten, …).

---

## Lokale Entwicklung

```bash
# Server (ohne Docker)
cd server
pip install fastapi uvicorn websockets
uvicorn main:app --host 0.0.0.0 --port 3001

# Client
cd client
pip install -r requirements.txt
python main.py
```

Server-Konstanten in `server/game.py`:

| Konstante | Wert | Bedeutung |
|-----------|------|-----------|
| `TICK_RATE` | 60 Hz | Physik-Update-Rate |
| `BROADCAST_RATE` | 20 Hz | State-Senderate |
| `SHIP_SPEED` | 220 px/s | Bewegungsgeschwindigkeit |
| `BULLET_SPEED` | 620 px/s | Geschossgeschwindigkeit |
| `SHOOT_COOLDOWN` | 0.5 s | Schussverzögerung |
| `SHIP_RADIUS` | 16 px | Kollisionsradius |
| `RESPAWN_DELAY` | 3 s | Respawn-Zeit nach Tod |
