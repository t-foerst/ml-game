"""PPO-Training gegen algorithmischen Gegner (1v1, mehrere Räume parallel).

Setup:
  1. Server starten: cd ../server && uvicorn main:app --port 3001
     Für schnelleres Training: SPEED_MULTIPLIER=10 uvicorn main:app --port 3001
  2. pip install -r requirements.txt
  3. python train.py

Umgebungsvariablen:
  N_ENVS=4          Anzahl paralleler Räume (Standard: 4)
  TOTAL_STEPS=300000
  SERVER_URL=ws://localhost:3001/ws

TensorBoard:
  tensorboard --logdir tb_logs
"""

import os
import socket
import sys
import time
from pathlib import Path

from env import MlGameEnv
from opponent import AlgorithmicBot
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

SERVER_URL  = os.environ.get("SERVER_URL",   "ws://localhost:3001/ws")
TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", "300_000"))
N_ENVS      = int(os.environ.get("N_ENVS",      "4"))

CHECKPOINT_DIR = Path("checkpoints")
TB_LOG_DIR     = Path("tb_logs")

# Gesamter Rollout bleibt ~2048 Steps unabhängig von N_ENVS
ROLLOUT_TOTAL = 2048
N_STEPS = max(64, ROLLOUT_TOTAL // N_ENVS)


def _check_server(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def make_env_fn(server_url: str, room: str, max_steps: int = 300):
    """Factory für SubprocVecEnv — muss picklebar sein (keine Closures über nicht-pickelbare Objekte)."""
    def _init():
        return MlGameEnv(server_url=server_url, room=room, max_steps=max_steps)
    return _init


def main() -> None:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    TB_LOG_DIR.mkdir(exist_ok=True)

    rooms = [f"training-{i}" for i in range(N_ENVS)]
    print(f"Starte Training mit {N_ENVS} Räumen: {rooms}")
    print(f"n_steps={N_STEPS} pro Env → {N_STEPS * N_ENVS} Steps pro Update-Runde")

    # Frühzeitig prüfen ob der Server erreichbar ist
    import urllib.parse
    parsed = urllib.parse.urlparse(SERVER_URL.replace("ws://", "http://").replace("wss://", "https://"))
    host = parsed.hostname or "localhost"
    port = parsed.port or 3001
    if not _check_server(host, port):
        print(f"\nFEHLER: Server nicht erreichbar auf {host}:{port}")
        print("Starte den Server zuerst:")
        print("  cd ../server && uvicorn main:app --port 3001")
        print("  (schnell: SPEED_MULTIPLIER=10 uvicorn main:app --port 3001)")
        sys.exit(1)

    # Einen Gegner-Bot pro Raum starten
    opponents = [AlgorithmicBot(SERVER_URL, room=r) for r in rooms]
    for opp in opponents:
        opp.start()
    time.sleep(1.5)  # Bots Zeit geben sich zu verbinden

    vec_env = SubprocVecEnv([
        make_env_fn(SERVER_URL, room=r, max_steps=300)
        for r in rooms
    ])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=N_STEPS,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device="auto",
        tensorboard_log=str(TB_LOG_DIR),
        policy_kwargs=dict(net_arch=[64, 64]),
    )

    try:
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=CheckpointCallback(
                save_freq=10_000,
                save_path=str(CHECKPOINT_DIR),
                name_prefix="ppo",
                verbose=1,
            ),
            tb_log_name="ppo",
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nAbgebrochen.")
    finally:
        model.save(str(CHECKPOINT_DIR / "latest"))
        print("Modell gespeichert → checkpoints/latest")
        for opp in opponents:
            opp.stop()
        vec_env.close()


if __name__ == "__main__":
    main()
