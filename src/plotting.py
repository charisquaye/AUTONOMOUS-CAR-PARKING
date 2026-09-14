"""
Generate training curves (mean ± spread across seeds) and evaluation bar plots.
All figures are produced from the raw .npz / .json logs committed to the repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def load_seed_returns(log_dir: Path) -> Dict[int, np.ndarray]:
    """Load episode returns for every seed_* subdirectory."""
    data = {}
    for seed_path in sorted(log_dir.glob("seed_*")):
        npz = seed_path / "episode_returns.npz"
        if npz.exists():
            seed = int(seed_path.name.split("_")[1])
            arr = np.load(npz)
            data[seed] = arr["returns"]
    return data


def smooth(x: np.ndarray, window: int = 20) -> np.ndarray:
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_training_curves(
    log_dir: Path,
    out_path: Path,
    window: int = 25,
    max_episodes: Optional[int] = None,
):
    """Mean episode return ± 1 std across seeds."""
    seed_returns = load_seed_returns(log_dir)
    if not seed_returns:
        print("No episode_returns.npz found – skipping training curve.")
        return

    # Align to the shortest run
    min_len = min(len(r) for r in seed_returns.values())
    if max_episodes:
        min_len = min(min_len, max_episodes)

    matrix = np.stack([r[:min_len] for r in seed_returns.values()], axis=0)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)

    mean_s = smooth(mean, window)
    std_s = smooth(std, window)
    x = np.arange(len(mean_s))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, mean_s, color="C0", label="Mean across seeds")
    ax.fill_between(x, mean_s - std_s, mean_s + std_s, color="C0", alpha=0.25, label="±1 std")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return")
    ax.set_title("PPO Training Return (parking-v0)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved training curve → {out_path}")


def plot_eval_comparison(results_path: Path, out_path: Path):
    """Bar plot of success rate and mean reward: PPO (aggregate) vs baseline."""
    with open(results_path) as f:
        data = json.load(f)

    agg = data.get("aggregate", {})
    base = data.get("baseline", {})

    labels = ["PPO (mean±std)", "Rule-based baseline"]
    success = [
        agg.get("ppo_mean_success_rate", 0.0),
        base.get("success_rate", 0.0),
    ]
    success_err = [
        agg.get("ppo_std_success_rate", 0.0),
        0.0,
    ]
    reward = [
        agg.get("ppo_mean_episode_reward", 0.0),
        base.get("mean_episode_reward", 0.0),
    ]
    reward_err = [
        agg.get("ppo_std_episode_reward", 0.0),
        base.get("std_episode_reward", 0.0),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(labels, success, yerr=success_err, capsize=6, color=["C0", "C1"])
    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Parking success rate")

    axes[1].bar(labels, reward, yerr=reward_err, capsize=6, color=["C0", "C1"])
    axes[1].set_ylabel("Mean episode return")
    axes[1].set_title("Episode return")

    fig.suptitle("Evaluation under identical protocol (30 episodes × seeds)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved evaluation comparison → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--eval-json", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="figures")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)

    plot_training_curves(log_dir, out_dir / "training_return.png")
    if args.eval_json:
        plot_eval_comparison(Path(args.eval_json), out_dir / "eval_comparison.png")


if __name__ == "__main__":
    main()
