# PPO-1: Autonomous Vehicle Parking  
**DSCD 614 – Reinforcement Learning Group Project**  
University of Ghana – 2025/2026

## Group Members
| Name | Student ID |
|------|------------|
| ADDO, Austine Gamey | 22424506 |
| Andrews Anseiku Junior | 22427819 |
| George Manuel | 22424752 |

**Project Option:** PPO-1  
**Algorithm:** Proximal Policy Optimisation (PPO)  
**Environment:** HighwayEnv `parking-v0`

---

## What is in this folder

```
PPO1_AUTONOMOUS_PARKING_FINAL/
├── src/                  # All source code (environment, baseline, train, evaluate, plot)
├── configs/              # Hyperparameter YAML (held constant across seeds)
├── docs/                 # Full MDP formulation with equations
├── scripts/              # Reproduction script
├── slides/               # Demonstration PowerPoint slides
├── submission/           # All Sakai submission documents
│   ├── Project_Report.md
│   ├── Hyperparameters_and_Seeds.md
│   ├── AI_Use_Declaration.md
│   └── Submission_Links.txt
├── requirements.txt
└── README.md
```

---

## Quick Start (Mac / Linux)

```bash
cd PPO1_AUTONOMOUS_PARKING_FINAL
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Full training (3 seeds)
python -m src.train --config configs/ppo_parking.yaml --seeds 42 123 456 --timesteps 300000 --n-envs 4
```

---

