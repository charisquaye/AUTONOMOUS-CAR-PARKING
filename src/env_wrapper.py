"""
Environment definition / wrapper for HighwayEnv parking-v0.

This module is original group work as required by the examination:
- Observation construction
- Action handling
- Termination / truncation logic
- Reward shaping (optional additional terms on top of the environment reward)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import highway_env
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import RecordEpisodeStatistics

# Ensure environments are registered
gym.register_envs(highway_env)


def make_parking_env(
    render_mode: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    flatten_obs: bool = True,
    max_episode_steps: int = 100,
) -> gym.Env:
    """
    Create a configured parking-v0 environment ready for PPO training.

    Parameters
    ----------
    render_mode : str or None
        "human", "rgb_array", or None.
    config_overrides : dict
        Extra keys merged into the environment config.
    flatten_obs : bool
        If True, convert the GoalEnv Dict observation into a flat Box
        suitable for standard PPO (observation + desired_goal).
    max_episode_steps : int
        Soft truncation horizon (in environment steps).
    """
    default_config = {
        "observation": {
            "type": "KinematicsGoal",
            "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
            "scales": [100, 100, 5, 5, 1, 1],
            "normalize": False,
        },
        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True,
            "dynamical": False,  # kinematic model is sufficient and faster
        },
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "duration": 20,  # seconds; interacts with policy_frequency
        "reward_weights": [1.0, 0.3, 0.0, 0.0, 0.02, 0.02],  # x,y,vx,vy,cos,sin
        "success_goal_reward": 0.12,
        "collision_reward": -5.0,
        "steering_range": np.deg2rad(45),
        "screen_width": 600,
        "screen_height": 300,
        "centering_position": [0.5, 0.5],
        "scaling": 7,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": render_mode != "human",
    }

    if config_overrides:
        # shallow merge for top-level; nested dicts need care
        for k, v in config_overrides.items():
            if isinstance(v, dict) and k in default_config and isinstance(default_config[k], dict):
                default_config[k].update(v)
            else:
                default_config[k] = v

    env = gym.make(
        "parking-v0",
        render_mode=render_mode,
        config=default_config,
    )

    # TimeLimit-style truncation (environment already has duration, but we enforce steps)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    if flatten_obs:
        env = FlattenGoalObs(env)

    env = RecordEpisodeStatistics(env)
    return env


class FlattenGoalObs(gym.ObservationWrapper):
    """
    Convert the GoalEnv Dict observation into a single flat Box:
        [observation (6), desired_goal (6)]  →  shape (12,)

    This makes the environment compatible with standard Stable-Baselines3 PPO
    without requiring HER or a custom policy that accepts Dict spaces.
    The achieved_goal is still available inside the original info / reward.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Dict)
        obs_space = env.observation_space["observation"]
        goal_space = env.observation_space["desired_goal"]
        low = np.concatenate([obs_space.low, goal_space.low]).astype(np.float32)
        high = np.concatenate([obs_space.high, goal_space.high]).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [observation["observation"], observation["desired_goal"]]
        ).astype(np.float32)


class ParkingRewardWrapper(gym.RewardWrapper):
    """
    Optional additional reward shaping on top of the native HighwayEnv reward.

    Native reward (from docs):
        R = -||s - s_g||_{W,p}^p  +  collision_term

    We keep the native term and can add a small success bonus or step cost
    if desired. By default this wrapper is a pass-through so that the
    environment reward remains the primary signal.
    """

    def __init__(self, env: gym.Env, success_bonus: float = 0.0, step_penalty: float = 0.0):
        super().__init__(env)
        self.success_bonus = success_bonus
        self.step_penalty = step_penalty

    def reward(self, reward: float) -> float:
        # The underlying env already returns the shaped proximity reward.
        # We only add optional terms here.
        info = self.env.unwrapped.info if hasattr(self.env.unwrapped, "info") else {}
        extra = 0.0
        if info.get("is_success", False):
            extra += self.success_bonus
        extra += self.step_penalty
        return float(reward + extra)


def get_env_info(env: gym.Env) -> Dict[str, Any]:
    """Utility to print a concise summary of spaces and config."""
    unwrapped = env.unwrapped
    return {
        "observation_space": str(env.observation_space),
        "action_space": str(env.action_space),
        "reward_weights": unwrapped.config.get("reward_weights"),
        "success_goal_reward": unwrapped.config.get("success_goal_reward"),
        "collision_reward": unwrapped.config.get("collision_reward"),
        "duration": unwrapped.config.get("duration"),
        "policy_frequency": unwrapped.config.get("policy_frequency"),
    }
