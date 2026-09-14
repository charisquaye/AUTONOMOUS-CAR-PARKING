"""
Evaluation harness for the trained PPO agent and the required rule-based baseline.

All metrics are computed under identical conditions (same seeds, same episodes,
deterministic policy, exploration disabled).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baseline import RuleBasedParkingController, evaluate_controller
from src.env_wrapper import make_parking_env


def evaluate_ppo(
    model_path: str | Path,
    n_episodes: int = 30,
    seed: int = 0,
    render: bool = False,
) -> Dict[str, float]:
    """Evaluate a saved PPO model under deterministic policy."""
    env = make_parking_env(
        render_mode="human" if render else None,
        flatten_obs=True,
        max_episode_steps=100,
    )
    model = PPO.load(str(model_path), env=env)

    successes, collisions, rewards, lengths = [], [], [], []
    final_distances, final_orients = [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        terminated = truncated = False
        ep_reward = 0.0
        steps = 0
        crashed = success = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            steps += 1
            if info.get("is_success", False):
                success = True
            if info.get("crashed", False):
                crashed = True

        if isinstance(obs, np.ndarray) and obs.shape[0] >= 12:
            dx = obs[6] - obs[0]
            dy = obs[7] - obs[1]
            final_distances.append(float(np.hypot(dx, dy)))
            cos_h, sin_h = obs[4], obs[5]
            gcos, gsin = obs[10], obs[11]
            orient_err = abs(np.arctan2(sin_h, cos_h) - np.arctan2(gsin, gcos))
            orient_err = min(orient_err, 2 * np.pi - orient_err)
            final_orients.append(float(orient_err))
        else:
            final_distances.append(np.nan)
            final_orients.append(np.nan)

        successes.append(float(success))
        collisions.append(float(crashed))
        rewards.append(ep_reward)
        lengths.append(steps)

    env.close()
    return {
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "mean_episode_reward": float(np.mean(rewards)),
        "std_episode_reward": float(np.std(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "mean_final_distance": float(np.nanmean(final_distances)),
        "mean_final_orientation_error": float(np.nanmean(final_orients)),
        "n_episodes": n_episodes,
        "seed_base": seed,
    }


def run_full_comparison(
    model_paths: Dict[int, str],
    n_eval_episodes: int = 30,
    eval_seed_base: int = 10_000,
) -> Dict[str, Any]:
    """
    Evaluate every trained seed and the rule-based baseline under the same
    evaluation seeds / episode count.
    """
    results = {"ppo_per_seed": {}, "baseline": None, "aggregate": {}}

    # Baseline (single deterministic controller)
    env = make_parking_env(flatten_obs=True, max_episode_steps=100)
    baseline = RuleBasedParkingController()
    baseline_metrics = evaluate_controller(
        baseline, env, n_episodes=n_eval_episodes, seed=eval_seed_base
    )
    results["baseline"] = baseline_metrics
    env.close()

    ppo_success, ppo_reward, ppo_collision = [], [], []

    for seed, path in model_paths.items():
        metrics = evaluate_ppo(
            path, n_episodes=n_eval_episodes, seed=eval_seed_base
        )
        results["ppo_per_seed"][seed] = metrics
        ppo_success.append(metrics["success_rate"])
        ppo_reward.append(metrics["mean_episode_reward"])
        ppo_collision.append(metrics["collision_rate"])

    results["aggregate"] = {
        "ppo_mean_success_rate": float(np.mean(ppo_success)),
        "ppo_std_success_rate": float(np.std(ppo_success)),
        "ppo_mean_episode_reward": float(np.mean(ppo_reward)),
        "ppo_std_episode_reward": float(np.std(ppo_reward)),
        "ppo_mean_collision_rate": float(np.mean(ppo_collision)),
        "baseline_success_rate": baseline_metrics["success_rate"],
        "baseline_mean_episode_reward": baseline_metrics["mean_episode_reward"],
        "n_seeds": len(model_paths),
        "n_eval_episodes_per_seed": n_eval_episodes,
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to a single .zip model")
    parser.add_argument("--models-dir", type=str, help="Directory containing seed_* subdirs")
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--out", type=str, default="results/evaluation.json")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.model:
        metrics = evaluate_ppo(args.model, args.n_episodes, args.seed, args.render)
        print(json.dumps(metrics, indent=2))
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)
    elif args.models_dir:
        models_dir = Path(args.models_dir)
        model_paths = {}
        for seed_dir in sorted(models_dir.glob("seed_*")):
            seed = int(seed_dir.name.split("_")[1])
            # Prefer best model, fall back to final
            best = seed_dir / "best" / "best_model.zip"
            final = seed_dir / "ppo_parking_final.zip"
            path = best if best.exists() else final
            if path.exists():
                model_paths[seed] = str(path)
        results = run_full_comparison(model_paths, args.n_episodes, args.seed)
        print(json.dumps(results, indent=2))
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
    else:
        # Baseline only
        env = make_parking_env(flatten_obs=True)
        metrics = evaluate_controller(
            RuleBasedParkingController(), env, n_episodes=args.n_episodes, seed=args.seed
        )
        print(json.dumps(metrics, indent=2))
        with open(args.out, "w") as f:
            json.dump({"baseline": metrics}, f, indent=2)


if __name__ == "__main__":
    main()
