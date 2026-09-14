# Markov Decision Process Formulation  
## PPO-1: Autonomous Vehicle Parking

**Course**: DSCD 614 – Reinforcement Learning  
**Group Project Option**: PPO-1  
**Environment**: HighwayEnv `parking-v0` (Gymnasium API)

---

### 1. State Space \(\mathcal{S}\)

The native observation of `parking-v0` is a GoalEnv dictionary:

\[
o_t = \bigl\{\text{observation},\ \text{achieved\_goal},\ \text{desired\_goal}\bigr\}
\]

Each component is a 6-dimensional continuous vector:

\[
\bigl[x,\ y,\ v_x,\ v_y,\ \cos h,\ \sin h\bigr]
\]

where \((x,y)\) is position (metres), \((v_x,v_y)\) velocity (m/s) and \(h\) the heading angle.  
Scales applied by the environment: \([100, 100, 5, 5, 1, 1]\).

**Implementation choice** (justified): we flatten the observation into a single 12-dimensional vector

\[
s_t = \bigl[\underbrace{x,y,v_x,v_y,\cos h,\sin h}_{\text{current}},\
\underbrace{g_x,g_y,g_{v_x},g_{v_y},\cos g,\sin g}_{\text{desired goal}}\bigr] \in \mathbb{R}^{12}
\]

so that a standard MLP policy can be used.  
No further normalisation is applied beyond the environment’s built-in scales.

**Markov property**: Under the kinematic bicycle model used by HighwayEnv the chosen state fully determines the next state given an action; the Markov property holds.

---

### 2. Action Space \(\mathcal{A}\)

Continuous control:

\[
a_t = \bigl[\delta,\ a\bigr] \in [-1,1]^2
\]

- \(\delta\): normalised steering angle (mapped to \(\pm 45^\circ\))  
- \(a\): normalised longitudinal acceleration (throttle / brake)

Action type: `ContinuousAction` with both longitudinal and lateral control enabled.

---

### 3. Reward Function

The native HighwayEnv reward for goal-conditioned tasks is a weighted \(p\)-norm:

\[
r_t = -\|s_t - s_g\|_{W,p}^{p} + c\cdot\mathbb{I}_{\text{collision}}
\]

with default weights \(W = [1.0,\ 0.3,\ 0,\ 0,\ 0.02,\ 0.02]\), \(p=0.5\) and collision penalty \(c=-5\).

In mathematical form (component-wise):

\[
r_t = -\Bigl(w_x|x-g_x|^p + w_y|y-g_y|^p + w_h\bigl(|\cos h-\cos g|^p + |\sin h-\sin g|^p\bigr)\Bigr)
+ c\cdot\mathbb{I}_{\text{crashed}}
\]

A success bonus of \(+0.12\) is awarded when the agent reaches the goal region (already present in the environment).  
We do **not** add extra dense shaping beyond the environment reward; any additional terms are declared in the configuration file.

---

### 4. Episode Termination and Truncation

- **Termination** (`terminated=True`):  
  - Vehicle collides with an obstacle or another vehicle, **or**  
  - Goal is reached (`is_success=True`).

- **Truncation** (`truncated=True`):  
  - Maximum episode length of 100 environment steps (≈ 20 s of simulated time at the chosen policy frequency), **or**  
  - Environment internal duration limit.

Termination and truncation are kept strictly separate as required by the Gymnasium API.

---

### 5. Discount Factor \(\gamma\)

\[
\gamma = 0.99
\]

**Justification**: Effective horizon of a parking manoeuvre is short (typically 5–15 s). With policy frequency 5 Hz a discount of 0.99 yields an effective horizon of roughly \(1/(1-\gamma)\approx 100\) steps, which comfortably covers the longest reasonable parking trajectories while still prioritising near-term success.

---

### 6. Transition Dynamics

The underlying dynamics are the kinematic bicycle model implemented inside HighwayEnv (deterministic given the action and current state). Stochasticity arises only from:

- random initial pose of the ego-vehicle,  
- random goal location (parking spot),  
- optional presence of static parked vehicles (configurable).

Hence the process is a finite-horizon MDP with continuous state and action spaces.

---

### 7. Summary Table

| Element              | Specification                                      |
|----------------------|----------------------------------------------------|
| State dimension      | 12 (flattened)                                     |
| Action dimension     | 2 continuous                                       |
| Reward               | weighted \(p\)-norm + collision penalty (eq. above)|
| \(\gamma\)           | 0.99                                               |
| Max episode steps    | 100                                                |
| Markov?              | Yes (kinematic state is sufficient)                |
| Algorithm            | PPO (Stable-Baselines3)                            |
| Required baseline    | Rule-based geometric controller                    |
