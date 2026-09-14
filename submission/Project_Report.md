# Autonomous Vehicle Parking using Proximal Policy Optimisation

**DSCD 614 – Reinforcement Learning**  
University of Ghana, Department of Computer Science  
MPhil/MSc Computer Science & Data Science, 2025/2026

**Group Members**  
- ADDO, Austine Gamey (22424506)  
- Andrews Anseiku Junior (22427819)  
- George Manuel (22424752)

**Project Option:** PPO-1  
**Algorithm:** Proximal Policy Optimisation (PPO)  
**Environment:** HighwayEnv `parking-v0`

------------------------------------------------------------------------

## 1. Introduction

Autonomous parking is a practical and safety-critical sub-task of autonomous driving. A vehicle must navigate from an arbitrary initial pose into a designated parking bay while respecting kinematic constraints and avoiding collisions. The problem is naturally formulated as a continuous-control Markov Decision Process (MDP) and is therefore well-suited to policy-gradient methods such as Proximal Policy Optimisation (PPO).

This project implements and evaluates a PPO agent on the HighwayEnv parking environment. The agent is compared against a deterministic rule-based geometric controller under a controlled multi-seed experimental protocol. The emphasis of the work is on rigorous formulation, reproducible protocol and honest analysis rather than on achieving published benchmark scores.

**Aims**  
1. Formulate the parking task as an MDP with explicit state, action, reward, termination and discount factor.  
2. Train a PPO agent with three independent random seeds.  
3. Evaluate the trained agent against the required rule-based baseline under identical conditions.  
4. Analyse convergence, stability and the effect of the chosen reward design.

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
- Total timesteps per seed: 300 000.
- Parallel environments: 4.

### 4.3 Baseline

A deterministic geometric controller is implemented:

1. Compute heading error toward the goal position.
2. Steer proportionally and accelerate (or reverse if pointing away).
3. When close to the goal, switch to orientation alignment and braking.

The baseline is evaluated with the identical evaluation harness, seeds and metric code used for the PPO agent.

### 4.4 Evaluation Protocol

- Held-out evaluation seeds.
- Exploration disabled (`deterministic=True`).
- 30 episodes per seed.
- Metrics: success rate, collision rate, mean episode return, mean final distance, mean final orientation error, mean episode length.
- Mean and standard deviation across seeds are reported for every metric.

------------------------------------------------------------------------

## 5. Results

All figures and tables below are produced from the raw logs and evaluation files committed to the repository (`logs/run_20260914_092903/` and `results/evaluation.json`).

### 5.1 Training Curves

Training was performed for 300 000 timesteps on each of three seeds (42, 123, 456). Approximately 17 800–18 200 episodes were collected per seed. The mean return of the last 100 episodes of each run remained negative and clustered tightly around −10.6 to −11.1:

| Seed | Episodes logged | Mean return (last 100) |
|------|-----------------|------------------------|
| 42   | 17 798          | −10.97                 |
| 123  | 17 978          | −11.06                 |
| 456  | 18 238          | −10.62                 |

The training-return plot (Figure 1, `figures/training_return.png`) shows that learning curves across the three seeds are highly similar: returns rise modestly from the initial random-policy regime but plateau at a still-negative value. No seed exhibits clear divergence or catastrophic collapse, indicating that the optimisation itself is stable even though final task performance remains low.

### 5.2 Evaluation against Baseline

Each trained model and the rule-based baseline were evaluated for 30 deterministic episodes. Aggregate statistics (mean ± standard deviation across the three training seeds) are reported below.

| Metric                         | PPO (mean ± std)          | Rule-based baseline      |
|--------------------------------|---------------------------|--------------------------|
| Success rate                   | 0.022 ± 0.031             | 0.000                    |
| Collision rate                 | 0.978 ± 0.031             | 0.833                    |
| Mean episode return            | −11.08 ± 0.11             | −28.29                   |
| Mean final distance            | 0.087 ± 0.015             | 0.301                    |
| Mean final orientation error   | 1.455 ± 0.223             | 1.445                    |
| Mean episode length            | 15.3 ± 0.5                | 45.6                     |

Per-seed PPO results:

| Seed | Success rate | Collision rate | Mean return | Mean final distance | Mean orient. error |
|------|--------------|----------------|-------------|---------------------|--------------------|
| 42   | 0.067        | 0.933          | −10.93      | 0.070               | 1.156              |
| 123  | 0.000        | 1.000          | −11.12      | 0.085               | 1.511              |
| 456  | 0.000        | 1.000          | −11.19      | 0.105               | 1.696              |

### 5.3 Statistical Note

With only three seeds the magnitude of any difference must be interpreted cautiously. PPO obtains a higher (less negative) mean return and a lower final distance than the baseline, yet both methods fail to park reliably. The observed success-rate difference (≈ 2 % versus 0 %) is smaller than the variation across seeds and cannot be regarded as statistically meaningful.

------------------------------------------------------------------------

## 6. Discussion

### 6.1 Convergence and Training Stability

The three independent training runs produced nearly identical learning curves and final returns. This indicates that the PPO optimisation is stable under the chosen hyperparameters (learning rate $3\times10^{-4}$, clip range 0.2, entropy coefficient 0.01). The absence of large variance across seeds is a positive finding; the algorithm is not brittle. However, the plateau at approximately −11 shows that the policy stops improving long before it masters the parking task.

### 6.2 Exploration

The entropy coefficient of 0.01 provides only mild stochasticity. In a continuous control problem whose successful trajectories occupy a narrow region of the action space (precise steering and braking near the goal), this level of exploration appears insufficient. Once the agent discovers that aggressive acceleration yields rapid collisions (and therefore large negative rewards), it tends to adopt a cautious but still unsuccessful policy that terminates early by collision or truncation.

### 6.3 Effect of Reward Design

The native HighwayEnv reward is a weighted $p$-norm proximity term plus a collision penalty of −5 and a small success bonus of +0.12. Because the collision penalty is large relative to the proximity reward, any trajectory that risks contact is heavily discouraged. In practice the agent learns to keep distance from obstacles but never acquires the fine-grained manoeuvres needed to enter the parking bay. The sparse success signal (+0.12) is received too rarely to shape the policy effectively within 300 000 steps.

### 6.4 Comparison with the Baseline

The geometric baseline also fails on the evaluation set (0 % success). Its longer episode lengths (≈ 46 steps versus ≈ 15 for PPO) show that it continues attempting alignment, whereas the learned policy aborts early. PPO nevertheless achieves a substantially higher mean return (−11 versus −28) and a smaller final distance to the goal, indicating that it has internalised a better proximity-seeking behaviour even though it does not complete the park.

### 6.5 Summary of Observed Limitations

- Success remains near zero for both methods under the chosen evaluation conditions.
- The combination of a strong collision penalty and limited exploration prevents the discovery of successful parking trajectories.
- 300 000 timesteps, while consistent across seeds, appear insufficient for reliable mastery of the continuous parking task with the current reward.

------------------------------------------------------------------------

## 7. Limitations and Deployment Considerations

- The simulator uses a simplified kinematic bicycle model; real vehicles have dynamics, tyre slip and actuator delays that are not modelled.
- The observation is fully observable and noise-free; a deployed system would require state estimation from sensors.
- Parking bays are static and free of pedestrians or moving vehicles.
- Compute was limited to the resources available to a student group within a fourteen-day window; longer training or larger networks may improve performance.
- The policy is deterministic at evaluation time; stochastic deployment would require additional safety layers.
- The present reward design strongly penalises collisions and provides only a weak success signal, which appears to hinder learning of complete parking manoeuvres.

------------------------------------------------------------------------

## 8. Conclusion and Further Work

We formulated autonomous parking as a continuous-control MDP, implemented a PPO agent and a rule-based baseline, and executed a reproducible multi-seed evaluation protocol. Training was stable across three seeds, yet final success rates remained near zero for both the learned policy and the geometric controller. The results highlight that a stable optimisation procedure alone is insufficient; reward design and exploration must be carefully tuned for the sparse-success continuous-control setting.

Further work includes:

- denser or hierarchical reward shaping that rewards intermediate progress toward a successful park,
- increased entropy or curiosity-driven exploration,
- longer training budgets or curriculum learning on progressively harder initial poses,
- addition of parked vehicles as obstacles and domain randomisation of vehicle parameters,
- transfer to a higher-fidelity simulator such as CARLA.

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
