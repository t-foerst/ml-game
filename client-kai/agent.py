"""PPO-Training für ml-game KI-Agent.

Start:   python agent.py
Setzt automatisch fort falls models/config.json vorhanden.
Strg+C   unterbricht sauber und speichert.
"""

import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from env import MLGameEnv

MODELS_DIR          = Path(__file__).parent / "models"
MODEL_FILE          = MODELS_DIR / "ppo_model"   # .zip wird von SB3 ergänzt
CONFIG_FILE         = MODELS_DIR / "config.json"
STEPS_PER_SESSION   = int(1e12)  # läuft bis Strg+C
SAVE_EVERY          = 10_000   # Zwischenspeicherung

DEFAULT_CONFIG: dict = {
    "learning_rate":          3e-4,
    "n_steps":                2048,
    "batch_size":             64,
    "n_epochs":               10,
    "gamma":                  0.99,
    "gae_lambda":             0.95,
    "clip_range":             0.2,
    "ent_coef":               0.01,
    "total_timesteps_trained": 0,
}


# ── Speichern / Laden ─────────────────────────────────────────────────────────

def save(model: PPO, config: dict) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    config["total_timesteps_trained"] = model.num_timesteps
    model.save(str(MODEL_FILE))
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[agent] Gespeichert — {config['total_timesteps_trained']:,} Steps trainiert.")


def load() -> tuple[PPO | None, dict]:
    if not (MODEL_FILE.with_suffix(".zip").exists() and CONFIG_FILE.exists()):
        print("[agent] Kein Speicherstand — starte neu.")
        return None, DEFAULT_CONFIG.copy()

    with open(CONFIG_FILE) as f:
        config = json.load(f)

    env   = MLGameEnv()
    model = PPO.load(
        str(MODEL_FILE), env=env,
        learning_rate = config["learning_rate"],
        n_steps       = config["n_steps"],
        batch_size    = config["batch_size"],
        n_epochs      = config["n_epochs"],
        gamma         = config["gamma"],
        gae_lambda    = config["gae_lambda"],
        clip_range    = config["clip_range"],
        ent_coef      = config["ent_coef"],
        verbose       = 0,
    )
    print(f"[agent] Geladen — {config['total_timesteps_trained']:,} Steps trainiert.")
    return model, config


# ── Callback: Zwischenspeicherung ─────────────────────────────────────────────

class _SaveCallback(BaseCallback):
    def __init__(self, config: dict, save_every: int):
        super().__init__()
        self._config     = config
        self._save_every = save_every
        self._next_save  = save_every

    def _on_step(self) -> bool:
        if self.model.num_timesteps >= self._next_save:
            self._next_save += self._save_every
            save(self.model, self._config)
        return True


# ── Training ──────────────────────────────────────────────────────────────────

def train() -> None:
    model, config = load()
    env = MLGameEnv(debug=True)

    if model is None:
        model = PPO(
            "MlpPolicy", env,
            learning_rate = config["learning_rate"],
            n_steps       = config["n_steps"],
            batch_size    = config["batch_size"],
            n_epochs      = config["n_epochs"],
            gamma         = config["gamma"],
            gae_lambda    = config["gae_lambda"],
            clip_range    = config["clip_range"],
            ent_coef      = config["ent_coef"],
            verbose       = 0,
        )
    else:
        model.set_env(env)

    print(f"[agent] Training startet — Ziel: +{STEPS_PER_SESSION:,} Steps")
    callback = _SaveCallback(config, SAVE_EVERY)

    try:
        model.learn(
            total_timesteps    = STEPS_PER_SESSION,
            callback           = callback,
            reset_num_timesteps= False,
        )
    except KeyboardInterrupt:
        print("\n[agent] Unterbrochen —", end=" ")

    save(model, config)


if __name__ == "__main__":
    train()
