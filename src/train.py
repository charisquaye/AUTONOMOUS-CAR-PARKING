"""
Training entry point for PPO on the parking environment.

Runs multiple independent seeds, logs to TensorBoard and CSV,
saves models and raw returns for later figure generation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.env_wrapper import make_parking_env


class EpisodeReturnLogger(BaseCallback):
    """Log episode returns and lengths to a list for later aggregation."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.returns: List[float] = []
        self.lengths: List[int] = []
        self.timesteps: List[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.returns.append(float(info["episode"]["r"]))
                self.lengths.append(int(info["episode"]["l"]))
                self.timesteps.append(self.num_timesteps)
        return True


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env_fn(seed: int, rank: int = 0, log_dir: str | None = None):
    def _init():
        env = make_parking_env(render_mode=None, flatten_obs=True, max_episode_steps=100)
        env.reset(seed=seed + rank)
        if log_dir:
            env = Monitor(env, filename=os.path.join(log_dir, f"monitor_{rank}"))
        return env
    return _init


def train_one_seed(
    seed: int,
    total_timesteps: int,
    hyperparams: Dict[str, Any],
    log_dir: Path,
    model_dir: Path,
    n_envs: int = 4,
) -> Dict[str, Any]:
    """Train a single PPO agent for the given seed and return summary stats."""
    seed_log_dir = log_dir / f"seed_{seed}"
    seed_model_dir = model_dir / f"seed_{seed}"
    seed_log_dir.mkdir(parents=True, exist_ok=True)
    seed_model_dir.mkdir(parents=True, exist_ok=True)

    # Vectorised environment
    if n_envs > 1:
        env = SubprocVecEnv([make_env_fn(seed, i, str(seed_log_dir)) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env_fn(seed, 0, str(seed_log_dir))])

    eval_env = DummyVecEnv([make_env_fn(seed + 1000, 0)])

    policy_kwargs = dict(hyperparams.get("policy_kwargs", {}))
    # Resolve activation function string to the actual torch.nn class
    if "activation_fn" in policy_kwargs and isinstance(policy_kwargs["activation_fn"], str):
        import torch.nn as nn
        name = policy_kwargs["activation_fn"].split(".")[-1]
        policy_kwargs["activation_fn"] = getattr(nn, name)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=hyperparams.get("learning_rate", 3e-4),
        n_steps=hyperparams.get("n_steps", 2048),
        batch_size=hyperparams.get("batch_size", 64),
        n_epochs=hyperparams.get("n_epochs", 10),
        gamma=hyperparams.get("gamma", 0.99),
        gae_lambda=hyperparams.get("gae_lambda", 0.95),
        clip_range=hyperparams.get("clip_range", 0.2),
        ent_coef=hyperparams.get("ent_coef", 0.01),
        vf_coef=hyperparams.get("vf_coef", 0.5),
        max_grad_norm=hyperparams.get("max_grad_norm", 0.5),
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        tensorboard_log=None,  # set to str(seed_log_dir / "tb") for local TensorBoard
        device="auto",
    )

    return_logger = EpisodeReturnLogger()
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=str(seed_model_dir),
        name_prefix="ppo_parking",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(seed_model_dir / "best"),
        log_path=str(seed_log_dir / "eval"),
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[return_logger, checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    # Final save
    final_path = seed_model_dir / "ppo_parking_final.zip"
    model.save(str(final_path))

    # Persist raw returns for figure generation
    np.savez(
        seed_log_dir / "episode_returns.npz",
        returns=np.array(return_logger.returns),
        lengths=np.array(return_logger.lengths),
        timesteps=np.array(return_logger.timesteps),
    )

    env.close()
    eval_env.close()

    return {
        "seed": seed,
        "final_model": str(final_path),
        "n_episodes_logged": len(return_logger.returns),
        "mean_last_100_return": float(np.mean(return_logger.returns[-100:]))
        if return_logger.returns
        else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description="Train PPO on parking-v0 (multi-seed)")
    parser.add_argument("--config", type=str, default="configs/ppo_parking.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--model-dir", type=str, default="models")
    args = parser.parse_args()

    cfg = load_config(args.config)
    hyperparams = cfg.get("hyperparameters", {})
    total_timesteps = args.timesteps or cfg.get("total_timesteps", 300_000)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = Path(args.log_dir) / f"run_{timestamp}"
    run_model_dir = Path(args.model_dir) / f"run_{timestamp}"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_model_dir.mkdir(parents=True, exist_ok=True)

    # Save the exact config used
    with open(run_log_dir / "config_used.yaml", "w") as f:
        yaml.dump({"hyperparameters": hyperparams, "total_timesteps": total_timesteps, "seeds": args.seeds}, f)

    summaries = []
    for seed in args.seeds:
        print(f"\n========== Training seed {seed} ==========")
        summary = train_one_seed(
            seed=seed,
            total_timesteps=total_timesteps,
            hyperparams=hyperparams,
            log_dir=run_log_dir,
            model_dir=run_model_dir,
            n_envs=args.n_envs,
        )
        summaries.append(summary)
        print(f"Seed {seed} finished. Mean last-100 return: {summary['mean_last_100_return']:.3f}")

    with open(run_log_dir / "training_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    print("\nAll seeds completed.")
    print(f"Logs  → {run_log_dir}")
    print(f"Models → {run_model_dir}")


if __name__ == "__main__":
    main()
