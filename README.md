# PPO-1: Autonomous Vehicle Parking

**DSCD 614 – Reinforcement Learning Group Project**  
University of Ghana – 2025/2026

## Group Members

| Name                    | Student ID |
|-------------------------|------------|
| ADDO, Austine Gamey     | 22424506   |
| George Manuel           | 22424752   |

**Project Option:** PPO-1  
**Algorithm:** Proximal Policy Optimisation (PPO)  
**Environment:** HighwayEnv `parking-v0`

---

## Repository Structure

```
├── src/                  # Source code
│   ├── env_wrapper.py    # Observation flattening wrapper
│   ├── baseline.py       # Rule-based geometric controller
│   ├── train.py          # PPO training entry point
│   ├── evaluate.py       # Evaluation harness
│   └── plotting.py       # Figure generation from logs
├── configs/              # Hyperparameter YAML (constant across seeds)
├── docs/                 # Full MDP formulation
├── scripts/              # Reproduction script
├── logs/                 # Raw training logs (committed)
├── models/               # Saved model weights (committed)
├── figures/              # Generated figures
├── results/              # Evaluation JSON
├── submission/           # Report and submission documents
├── requirements.txt      # Pinned dependencies
└── README.md
```

---

## Installation

```bash
git clone https://github.com/charisquaye/AUTONOMOUS-CAR-PARKING.git
cd AUTONOMOUS-CAR-PARKING

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Training

Train PPO on three seeds (42, 123, 456) for 300 000 timesteps each:

```bash
python -m src.train \
  --config configs/ppo_parking.yaml \
  --seeds 42 123 456 \
  --timesteps 300000 \
  --n-envs 4
```

Logs and models are written under `logs/run_*` and `models/run_*`.

---

## Evaluation

Evaluate the trained models and the rule-based baseline:

```bash
python -m src.evaluate \
  --models-dir models/run_20260914_092903 \
  --n-episodes 30 \
  --out results/evaluation.json
```

---

## Reproduce Figures

Regenerate the exact figures used in the report from the committed logs:

```bash
python -m src.plotting \
  --log-dir logs/run_20260914_092903 \
  --eval-json results/evaluation.json \
  --out-dir figures
```

- `figures/training_return.png` – training curves across seeds  
- `figures/eval_comparison.png` – PPO vs baseline metrics  

---

## Single-Command Headline Reproduction

```bash
bash scripts/reproduce_headline.sh
```

(Uses a shorter timestep budget by default for quick checks; set `TIMESTEPS=300000` for the full protocol.)

---

## Key Results (from committed evaluation)

| Metric              | PPO (mean ± std) | Baseline |
|---------------------|------------------|----------|
| Success rate        | 0.022 ± 0.031    | 0.000    |
| Collision rate      | 0.978 ± 0.031    | 0.833    |
| Mean episode return | −11.08 ± 0.11    | −28.29   |

Full analysis is in `submission/Project_Report.md` (and the PDF version).

---

## Citation / Commit

Final submission commit SHA is recorded in `submission/Submission_Links.txt`.
