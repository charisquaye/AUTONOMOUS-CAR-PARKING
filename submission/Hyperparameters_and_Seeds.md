# Hyperparameters and Seeds

**Project:** PPO-1 Autonomous Vehicle Parking  
**Group Members:** ADDO Austine Gamey (22424506), George Manuel (22424752)

---

## 1. Random Seeds

| Purpose | Seeds |
|---------|-------|
| Training | 42, 123, 456 |
| Evaluation (held-out) | 10000, 10001, 10002 |

All training seeds are held constant across hyperparameter settings. Hyperparameters are identical for every seed.

---

## 2. Full Hyperparameter Table (PPO – Stable-Baselines3)

These values are taken from `configs/ppo_parking.yaml` and were held constant across all three training seeds.

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Algorithm | PPO (Stable-Baselines3) | |
| Policy | MlpPolicy | |
| Network architecture (actor) | [256, 256] | |
| Network architecture (critic) | [256, 256] | |
| Activation function | ReLU | |
| Learning rate | 3 × 10⁻⁴ | |
| n_steps | 2048 | |
| batch_size | 64 | |
| n_epochs | 10 | |
| Discount factor γ | 0.99 | Justified by short effective horizon of parking manoeuvres |
| GAE λ | 0.95 | |
| Clip range | 0.2 | |
| Entropy coefficient | 0.01 | Mild exploration |
| Value function coefficient | 0.5 | |
| Max gradient norm | 0.5 | |
| Total timesteps per seed | 300 000 | May be reduced if compute is limited (state constraint in report) |
| Number of parallel environments | 4 | |
| Device | auto (CPU/GPU) | |

Any hyperparameter not listed above uses the Stable-Baselines3 default.

---

## 3. Environment Configuration

| Parameter | Value |
|-----------|-------|
| Environment ID | parking-v0 (HighwayEnv) |
| Observation | Flattened KinematicsGoal (12-dimensional) |
| Action | ContinuousAction (steering + acceleration) ∈ [-1, 1]² |
| Max episode steps | 100 |
| Reward weights | [1.0, 0.3, 0.0, 0.0, 0.02, 0.02] |
| Collision reward | −5.0 |
| Success goal reward | 0.12 |
| Policy frequency | 5 Hz |
| Simulation frequency | 15 Hz |

---

## 4. Evaluation Protocol

- Exploration disabled (`deterministic=True`)
- ≥ 30 evaluation episodes per seed
- Same evaluation seeds and metric code used for both the PPO agent and the rule-based baseline
- Mean and standard deviation reported across seeds for every metric

---

*This document is part of the formal submission package required by DSCD 614.*
