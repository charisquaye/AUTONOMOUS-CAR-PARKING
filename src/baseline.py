"""
Rule-based parking controller (required baseline for PPO-1).

This is original group work. The controller uses a simple geometric heuristic:
1. Align the vehicle heading roughly toward the goal.
2. Drive forward / reverse while reducing distance.
3. Apply a final orientation correction when close.

It is deliberately simple so that a well-trained PPO agent can demonstrably
outperform it under identical evaluation conditions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import spaces


class RuleBasedParkingController:
    """
    Deterministic rule-based baseline for the continuous parking task.

    Action = [steering, acceleration] both in [-1, 1].
    """

    def __init__(
        self,
        steering_gain: float = 1.5,
        accel_gain: float = 0.8,
        goal_distance_threshold: float = 0.15,  # normalised units (scales ~100 m)
        heading_tolerance: float = 0.25,  # rad
    ):
        self.steering_gain = steering_gain
        self.accel_gain = accel_gain
        self.goal_distance_threshold = goal_distance_threshold
        self.heading_tolerance = heading_tolerance

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Produce an action given a flattened observation of shape (12,):
            [x, y, vx, vy, cos_h, sin_h,  gx, gy, gvx, gvy, gcos, gsin]
        """
        obs = observation.astype(np.float64)
        # Current state (first 6)
        x, y, vx, vy, cos_h, sin_h = obs[:6]
        # Desired goal (last 6)
        gx, gy, _, _, gcos, gsin = obs[6:]

        # Position error (in the scaled observation coordinates)
        dx = gx - x
        dy = gy - y
        dist = np.hypot(dx, dy)

        # Current heading angle
        heading = np.arctan2(sin_h, cos_h)
        # Desired heading from goal orientation
        desired_heading = np.arctan2(gsin, gcos)

        # Angle to the goal position
        angle_to_goal = np.arctan2(dy, dx)
        # Heading error relative to the direction toward the goal
        heading_error_to_goal = self._angle_diff(angle_to_goal, heading)
        # Final orientation error
        orientation_error = self._angle_diff(desired_heading, heading)

        # Simple two-phase controller
        if dist > self.goal_distance_threshold:
            # Approach phase: steer toward goal, accelerate proportional to distance
            steering = np.clip(self.steering_gain * heading_error_to_goal, -1.0, 1.0)
            # Prefer forward motion; reverse only if we are pointing the wrong way
            forward = 1.0 if abs(heading_error_to_goal) < np.pi / 2 else -0.6
            acceleration = np.clip(self.accel_gain * forward * min(dist * 2.0, 1.0), -1.0, 1.0)
        else:
            # Alignment phase: reduce orientation error, slow down
            steering = np.clip(self.steering_gain * orientation_error, -1.0, 1.0)
            acceleration = np.clip(-0.4 * np.sign(vx * cos_h + vy * sin_h), -1.0, 1.0)  # brake

        action = np.array([steering, acceleration], dtype=np.float32)
        return action, None

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Smallest signed difference a - b in [-pi, pi]."""
        d = a - b
        return (d + np.pi) % (2 * np.pi) - np.pi


class RandomController:
    """Uniform random continuous actions (additional weak baseline)."""

    def __init__(self, action_space: spaces.Box):
        self.action_space = action_space

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        return self.action_space.sample(), None


def evaluate_controller(
    controller,
    env,
    n_episodes: int = 30,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Run a controller for a fixed number of episodes and collect metrics
    identical to those used for the PPO agent.
    """
    successes = []
    collisions = []
    episode_rewards = []
    episode_lengths = []
    final_distances = []
    final_orientation_errors = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep if seed is not None else None)
        terminated = truncated = False
        ep_reward = 0.0
        steps = 0
        crashed = False
        success = False

        while not (terminated or truncated):
            action, _ = controller.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            if info.get("is_success", False):
                success = True
            if info.get("crashed", False):
                crashed = True

        # Recover final pose error from the last observation if available
        if isinstance(obs, np.ndarray) and obs.shape[0] >= 12:
            dx = obs[6] - obs[0]
            dy = obs[7] - obs[1]
            final_distances.append(float(np.hypot(dx, dy)))
            cos_h, sin_h = obs[4], obs[5]
            gcos, gsin = obs[10], obs[11]
            orient_err = abs(np.arctan2(sin_h, cos_h) - np.arctan2(gsin, gcos))
            orient_err = min(orient_err, 2 * np.pi - orient_err)
            final_orientation_errors.append(float(orient_err))
        else:
            final_distances.append(np.nan)
            final_orientation_errors.append(np.nan)

        successes.append(float(success))
        collisions.append(float(crashed))
        episode_rewards.append(ep_reward)
        episode_lengths.append(steps)

    return {
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "std_episode_reward": float(np.std(episode_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_final_distance": float(np.nanmean(final_distances)),
        "mean_final_orientation_error": float(np.nanmean(final_orientation_errors)),
        "n_episodes": n_episodes,
    }
