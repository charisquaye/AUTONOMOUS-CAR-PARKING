# Autonomous Vehicle Parking using Proximal Policy Optimisation

**DSCD 614 – Reinforcement Learning**\
University of Ghana, Department of Computer Science\
MPhil/MSc Computer Science & Data Science, 2025/2026

**Group Members**\
- ADDO, Austine Gamey (22424506)\
- Andrews Anseiku Junior (22427819)\
- George Manuel (22424752)

**Project Option:** PPO-1\
**Algorithm:** Proximal Policy Optimisation (PPO)\
**Environment:** HighwayEnv `parking-v0`

------------------------------------------------------------------------

## 1. Introduction

Autonomous parking is a practical and safety-critical sub-task of autonomous driving. A vehicle must navigate from an arbitrary initial pose into a designated parking bay while respecting kinematic constraints and avoiding collisions. The problem is naturally formulated as a continuous-control Markov Decision Process (MDP) and is therefore well-suited to policy-gradient methods such as Proximal Policy Optimisation (PPO).

This project implements and evaluates a PPO agent on the HighwayEnv parking environment. The agent is compared against a deterministic rule-based geometric controller under a controlled multi-seed experimental protocol. The emphasis of the work is on rigorous formulation, reproducible protocol and honest analysis rather than on achieving published benchmark scores.

**Aims** 1. Formulate the parking task as an MDP with explicit state, action, reward, termination and discount factor. 2. Train a PPO agent with three independent random seeds. 3. Evaluate the trained agent against the required rule-based baseline under identical conditions. 4. Analyse convergence, stability and the effect of the chosen reward design.

------------------------------------------------------------------------

## 2. Background

### 2.1 Proximal Policy Optimisation

PPO (Schulman et al., 2017) is an on-policy actor-critic algorithm that constrains policy updates by a clipped surrogate objective. It offers a favourable trade-off between sample efficiency and implementation simplicity and is widely used for continuous-control problems. Stable-Baselines3 provides a well-tested implementation that we adopt; the assessed contribution therefore lies in the problem formulation, environment handling, baseline, experimental protocol and analysis.

### 2.2 Related Work

HighwayEnv (Leurent, 2018) supplies a lightweight kinematic simulator with a goal-conditioned parking task. Previous work has applied HER+SAC and other off-policy methods to the same environment. PPO remains attractive because of its stability and the modest computational budget available for a fourteen-day group project.

------------------------------------------------------------------------

## 3. Problem Formulation (MDP)

### 3.1 State Space

The native observation of `parking-v0` is a GoalEnv dictionary containing `observation`, `achieved_goal` and `desired_goal`. Each component is the six-dimensional kinematic vector

$$
[x,\ y,\ v_x,\ v_y,\ \cos h,\ \sin h].
$$

We flatten the observation into a single 12-dimensional continuous vector

$$
s_t = [\underbrace{x,y,v_x,v_y,\cos h,\sin h}_{\text{current}},\
\underbrace{g_x,g_y,g_{v_x},g_{v_y},\cos g,\sin g}_{\text{desired goal}}] \in \mathbb{R}^{12}.
$$

No additional normalisation is applied beyond the environment’s built-in scales. Under the kinematic bicycle model the chosen representation is Markovian.

### 3.2 Action Space

$$
a_t = [\delta,\ a] \in [-1,1]^2
$$

where $\delta$ is normalised steering angle and $a$ is normalised longitudinal acceleration.

### 3.3 Reward Function

The environment supplies a weighted $p$-norm proximity reward together with a collision penalty:

$$
r_t = -\|s_t - s_g\|_{W,p}^{p} + c\cdot\mathbb{I}_{\text{collision}},
$$

with default weights $W = [1.0,\ 0.3,\ 0,\ 0,\ 0.02,\ 0.02]$, $p = 0.5$ and $c = -5$. A success bonus of $+0.12$ is awarded when the goal region is reached. We retain this native reward without further dense shaping.

### 3.4 Termination and Truncation

- **Terminated** when the vehicle collides or the goal is reached (`is_success = True`).
- **Truncated** after 100 environment steps (approximately 20 s of simulated time).

### 3.5 Discount Factor

$$
\gamma = 0.99
$$

The effective horizon of a parking manoeuvre is short (typically 5–15 s). A discount of 0.99 yields an effective horizon of roughly 100 steps, matching the task length while still prioritising near-term success.

------------------------------------------------------------------------

## 4. Methodology

### 4.1 Environment Construction

We wrap HighwayEnv `parking-v0` with a custom observation flattener so that a standard MLP policy can be used. The Gymnasium API is employed throughout; the deprecated OpenAI Gym package is not used.

### 4.2 Network Architecture and Training

- Policy: MLP with two hidden layers of 256 units (actor and critic).
- Algorithm: PPO (Stable-Baselines3).
- Hyperparameters are listed in the accompanying `Hyperparameters_and_Seeds` document and are held constant across the three training seeds (42, 123, 456).
- Total timesteps per seed: 300 000 (or a reduced number if compute is limited; any reduction is stated in the results section).

### 4.3 Baseline

A deterministic geometric controller is implemented:

1.  Compute heading error toward the goal position.
2.  Steer proportionally and accelerate (or reverse if pointing away).
3.  When close to the goal, switch to orientation alignment and braking.

The baseline is evaluated with the identical evaluation harness, seeds and metric code used for the PPO agent.

### 4.4 Evaluation Protocol

- Held-out evaluation seeds.
- Exploration disabled (`deterministic=True`).
- At least 30 episodes per seed.
- Metrics: success rate, collision rate, mean episode return, mean final distance, mean final orientation error, mean episode length.
- Mean and standard deviation across seeds are reported for every metric.

------------------------------------------------------------------------

## 5. Results

*(After full training the following tables and figures will be populated from the committed logs.)*

### 5.1 Training Curves

Training return versus episode (mean ± 1 standard deviation across the three seeds) will be shown in Figure 1. Raw episode returns are stored as `.npz` files in the repository so that the figure can be regenerated exactly.

### 5.2 Evaluation against Baseline

| Metric                       | PPO (mean ± std) | Rule-based baseline |
|------------------------------|------------------|---------------------|
| Success rate                 | *to be filled*   | *to be filled*      |
| Mean episode return          | *to be filled*   | *to be filled*      |
| Collision rate               | *to be filled*   | *to be filled*      |
| Mean final distance          | *to be filled*   | *to be filled*      |
| Mean final orientation error | *to be filled*   | *to be filled*      |

### 5.3 Statistical Note

With only three seeds, claims about the magnitude of the difference between PPO and the baseline are necessarily cautious. We report the observed means and standard deviations and state whether the difference exceeds the variation across seeds.

------------------------------------------------------------------------

## 6. Discussion

*(To be completed after full training.)*

Topics that will be addressed:

- Convergence behaviour and training stability across seeds.
- Quality of exploration under the chosen entropy coefficient.
- Effect of the native weighted $p$-norm reward on final pose accuracy.
- Situations in which the rule-based baseline fails (large initial heading error, tight manoeuvres) and whether the learned policy overcomes them.
- Sensitivity to the success threshold and collision penalty.

------------------------------------------------------------------------

## 7. Limitations and Deployment Considerations

- The simulator uses a simplified kinematic bicycle model; real vehicles have dynamics, tyre slip and actuator delays that are not modelled.
- The observation is fully observable and noise-free; a deployed system would require state estimation from sensors.
- Parking bays are static and free of pedestrians or moving vehicles.
- Compute was limited to the resources available to a student group within a fourteen-day window; longer training or larger networks may improve performance.
- The policy is deterministic at evaluation time; stochastic deployment would require additional safety layers.

------------------------------------------------------------------------

## 8. Conclusion and Further Work

We have formulated autonomous parking as a continuous-control MDP, implemented a PPO agent and a rule-based baseline, and defined a reproducible multi-seed evaluation protocol. After full training the quantitative comparison will indicate whether the learned policy improves upon the geometric heuristic under the chosen metrics.

Further work includes the addition of parked vehicles as obstacles, domain randomisation of vehicle parameters, and transfer to a higher-fidelity simulator such as CARLA.

------------------------------------------------------------------------

## References

- Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
- Leurent, E. (2018). An Environment for Autonomous Driving Decision-Making. <https://github.com/Farama-Foundation/HighwayEnv>.
- Raffin, A., et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. JMLR.

------------------------------------------------------------------------

## Appendix

- Full hyperparameter table: see `Hyperparameters_and_Seeds.pdf`
- MDP equations: see `docs/MDP_FORMULATION.md` in the repository
- Raw experiment logs and model weights: committed to the public GitHub repository
