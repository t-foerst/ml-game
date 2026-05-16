"""PPO-Selbstspiel-Training für ml-game.

Setup:
  1. Server starten (im server/-Ordner mit docker compose up, oder lokal mit uvicorn)
  2. pip install -r requirements.txt
  3. python train.py

Für schnelleres Training: Server mit SPEED_MULTIPLIER=10 starten.
  cd ../server && SPEED_MULTIPLIER=10 uvicorn main:app --port 3001

Optionen (Umgebungsvariablen):
  SERVER_URL     ws://localhost:3001/ws
  N_ROOMS        8          (Anzahl paralleler Räume = paralleler Envs)
  TOTAL_STEPS    5000000    (Trainingsschritte gesamt)
  RESUME         1          (vorheriges Training fortsetzen)

TensorBoard:
  tensorboard --logdir tb_logs
"""

import os
import random
import sys
import time
from multiprocessing import freeze_support
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import MlGameEnv
from opponent import OpponentBot

# ── Konfiguration ─────────────────────────────────────────────────────────────

SERVER_URL     = os.environ.get("SERVER_URL",  "ws://localhost:3001/ws")
N_ROOMS        = int(os.environ.get("N_ROOMS",       "8"))
TOTAL_STEPS    = int(os.environ.get("TOTAL_STEPS",   "5_000_000"))
RESUME         = os.environ.get("RESUME", "0") == "1" or "--resume" in sys.argv

CHECKPOINT_DIR = Path("checkpoints")
TB_LOG_DIR     = Path("tb_logs")

SELFPLAY_FREQ  = 25_000   # Schritte zwischen Checkpoint-Saves für Selbstspiel
POOL_SIZE      = 8        # Maximale Checkpoints im Selbstspiel-Pool
MAX_STEPS_EP   = 2000     # Maximale Schritte pro Episode


# ── Selbstspiel-Callback ──────────────────────────────────────────────────────

class SelfPlayCallback(BaseCallback):
    """
    Speichert alle SELFPLAY_FREQ Schritte einen Checkpoint und aktualisiert
    die Gegner-Bots auf zufällige Versionen aus dem Pool.
    """

    def __init__(self, opponents: list[OpponentBot], save_dir: Path,
                 save_freq: int = SELFPLAY_FREQ, verbose: int = 1):
        super().__init__(verbose)
        self.opponents = opponents
        self.save_dir  = save_dir
        self.save_freq = save_freq
        self._pool:    list[str] = []

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True

        path = str(self.save_dir / f"selfplay_{self.n_calls:08d}")
        self.model.save(path)
        self._pool.append(path)
        if len(self._pool) > POOL_SIZE:
            self._pool.pop(0)

        for opp in self.opponents:
            ckpt = random.choice(self._pool)
            try:
                opp.update_model(PPO.load(ckpt, device="cpu"))
            except Exception as e:
                if self.verbose:
                    print(f"[SelfPlay] Ladefehler {ckpt}: {e}")

        if self.verbose:
            print(
                f"[SelfPlay] Schritt {self.n_calls:,} – "
                f"{len(self._pool)} Checkpoints im Pool"
            )
        return True


# ── Haupt-Training ────────────────────────────────────────────────────────────

def main() -> None:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    TB_LOG_DIR.mkdir(exist_ok=True)

    print(f"Server:      {SERVER_URL}")
    print(f"Räume:       {N_ROOMS}  (training-0 … training-{N_ROOMS - 1})")
    print(f"Schritte:    {TOTAL_STEPS:,}")
    print(f"Fortsetzen:  {RESUME}")

    # ── Gegner-Bots starten ───────────────────────────────────────────────────
    print("\nStarte Gegner-Bots (je einer pro Raum)…")
    opponents = [
        OpponentBot(SERVER_URL, room=f"training-{i}")
        for i in range(N_ROOMS)
    ]
    for opp in opponents:
        opp.start()
    time.sleep(2.0)   # Verbindung aufbauen lassen

    # ── Trainings-Envs ────────────────────────────────────────────────────────
    print("Erstelle Trainingsumgebungen…")

    def make_env(idx: int):
        def _init():
            return MlGameEnv(
                server_url=SERVER_URL,
                room=f"training-{idx}",
                max_steps=MAX_STEPS_EP,
            )
        return _init

    envs = SubprocVecEnv([make_env(i) for i in range(N_ROOMS)])

    # ── PPO-Modell ────────────────────────────────────────────────────────────
    latest = CHECKPOINT_DIR / "latest.zip"
    if RESUME and latest.exists():
        print(f"Lade Modell: {latest}")
        model = PPO.load(
            str(CHECKPOINT_DIR / "latest"),
            env=envs,
            device="auto",
            tensorboard_log=str(TB_LOG_DIR),
        )
    else:
        model = PPO(
            policy          = "MlpPolicy",
            env             = envs,
            learning_rate   = 3e-4,
            n_steps         = 512,       # Schritte pro Env zwischen Updates
            batch_size      = 256,
            n_epochs        = 10,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            ent_coef        = 0.01,      # Explorations-Bonus
            vf_coef         = 0.5,
            max_grad_norm   = 0.5,
            verbose         = 1,
            device          = "auto",
            tensorboard_log = str(TB_LOG_DIR),
            policy_kwargs   = dict(net_arch=[256, 256]),
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        SelfPlayCallback(opponents, CHECKPOINT_DIR, verbose=1),
        CheckpointCallback(
            save_freq   = 100_000,
            save_path   = str(CHECKPOINT_DIR),
            name_prefix = "ppo_mlgame",
            verbose     = 1,
        ),
    ]

    # ── Training ──────────────────────────────────────────────────────────────
    print("\nTraining startet – Ctrl+C zum Abbrechen\n")
    try:
        model.learn(
            total_timesteps     = TOTAL_STEPS,
            callback            = callbacks,
            reset_num_timesteps = not RESUME,
            tb_log_name         = "ppo_selfplay",
            progress_bar        = True,
        )
    except KeyboardInterrupt:
        print("\nAbgebrochen.")
    finally:
        model.save(str(CHECKPOINT_DIR / "latest"))
        print("Modell gespeichert → checkpoints/latest")
        for opp in opponents:
            opp.stop()
        envs.close()


if __name__ == "__main__":
    freeze_support()   # Windows-Kompatibilität für SubprocVecEnv
    main()
