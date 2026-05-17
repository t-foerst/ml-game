"""PPO-Training gegen algorithmischen Gegner (1v1, 1 Raum).

Setup:
  1. Server starten: cd ../server && uvicorn main:app --port 3001
     Für schnelleres Training: SPEED_MULTIPLIER=10 uvicorn main:app --port 3001
  2. pip install -r requirements.txt
  3. python train.py

TensorBoard:
  tensorboard --logdir tb_logs
"""

import os
import time
from pathlib import Path

from env import MlGameEnv
from opponent import AlgorithmicBot
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

SERVER_URL  = os.environ.get("SERVER_URL",   "ws://localhost:3001/ws")
TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", "300_000"))

CHECKPOINT_DIR = Path("checkpoints")
TB_LOG_DIR     = Path("tb_logs")


def main() -> None:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    TB_LOG_DIR.mkdir(exist_ok=True)

    opponent = AlgorithmicBot(SERVER_URL, room="training")
    opponent.start()
    time.sleep(1.0)

    env = MlGameEnv(server_url=SERVER_URL, room="training", max_steps=300)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,       # Samples pro Update-Runde
        batch_size=256,     # Mini-Batch-Größe (2048/256 = 8 Batches pro Epoch)
        n_epochs=4,         # Wenige Epochs = stabiler bei kleinem Datensatz
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,      # Exploration-Bonus
        verbose=1,
        device="auto",
        tensorboard_log=str(TB_LOG_DIR),
        policy_kwargs=dict(net_arch=[64, 64]),  # kleines Netz für einfache Aufgabe
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
        opponent.stop()
        env.close()


if __name__ == "__main__":
    main()
