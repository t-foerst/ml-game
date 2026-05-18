"""PPO-Ausweich-Training gegen algorithmischen Kreis-Gegner (mehrere Räume parallel).

Setup:
  1. Server starten: cd ../server && uvicorn main:app --port 3001
     Für schnelleres Training: SPEED_MULTIPLIER=10 uvicorn main:app --port 3001
  2. pip install -r requirements.txt
  3. python train.py

Umgebungsvariablen:
  N_ENVS=4          Anzahl paralleler Räume (Standard: 4)
  TOTAL_STEPS=500000
  SERVER_URL=ws://localhost:3001/ws
  SPEED_MULTIPLIER=1

TensorBoard:
  tensorboard --logdir tb_logs
"""

import os
import socket
import sys
import time
from pathlib import Path

from env import MlGameEnv
from opponent import CircleOpponent
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

SERVER_URL       = os.environ.get("SERVER_URL",       "ws://localhost:3001/ws")
TOTAL_STEPS      = int(os.environ.get("TOTAL_STEPS",  "500_000"))
N_ENVS           = int(os.environ.get("N_ENVS",       "4"))
SPEED_MULTIPLIER = float(os.environ.get("SPEED_MULTIPLIER", "1.0"))

CHECKPOINT_DIR = Path("checkpoints")
TB_LOG_DIR     = Path("tb_logs")

# Gesamter Rollout bleibt ~2048 Steps unabhängig von N_ENVS
N_STEPS = max(64, 2048 // N_ENVS)


def _check_server(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=3.0):
            return True
    except OSError:
        return False


def make_env_fn(server_url: str, room: str):
    def _init():
        return MlGameEnv(server_url=server_url, room=room, max_steps=1000)
    return _init


def main() -> None:
    import urllib.parse

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    TB_LOG_DIR.mkdir(exist_ok=True)

    parsed = urllib.parse.urlparse(SERVER_URL.replace("ws://", "http://"))
    host, port = parsed.hostname or "localhost", parsed.port or 3001
    if not _check_server(host, port):
        print(f"FEHLER: Server nicht erreichbar auf {host}:{port}")
        sys.exit(1)

    rooms     = [f"training-{i}" for i in range(N_ENVS)]
    opponents = [
        CircleOpponent(SERVER_URL, room=r, speed_multiplier=SPEED_MULTIPLIER)
        for r in rooms
    ]
    for opp in opponents:
        opp.start()
    time.sleep(1.5)

    vec_env = SubprocVecEnv([make_env_fn(SERVER_URL, r) for r in rooms])

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

    print(f"Räume: {N_ENVS}  |  n_steps/env: {N_STEPS}  |  Rollout: {N_STEPS * N_ENVS}")

    try:
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=CheckpointCallback(
                save_freq=50_000,
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
