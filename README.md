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

## Examination Deliverables Checklist

| Deliverable | Location / Status |
|-------------|-------------------|
| Code (GitHub) | `src/`, `configs/`, `docs/` – push to your public repo |
| Project Report | `submission/Project_Report.md` → convert to PDF |
| Hyperparameters & Seeds | `submission/Hyperparameters_and_Seeds.md` → convert to PDF |
| AI Use Declaration | `submission/AI_Use_Declaration.md` → convert to PDF |
| Submission_Links.txt | `submission/Submission_Links.txt` (names already filled) |
| Demonstration slides | `slides/PPO1_Demonstration_Slides.pptx` |
| YouTube video (≤ 20 min) | **You must record this** – all three members must appear |
| Multi-seed training logs | Run the training command above, then commit the logs |

---

## How to convert Markdown → PDF (Mac)

```bash
# Option A – using pandoc (recommended)
brew install pandoc
pandoc submission/Project_Report.md -o submission/Project_Report.pdf
pandoc submission/Hyperparameters_and_Seeds.md -o submission/Hyperparameters_and_Seeds.pdf
pandoc submission/AI_Use_Declaration.md -o submission/AI_Use_Declaration.pdf

# Option B – open in Word / Google Docs and export as PDF
```

---

## Important Notes

1. The report contains placeholders for numerical results. After you finish full training, fill in the tables in Section 5 and expand the Discussion.
2. Record the final Git commit SHA and the YouTube link in `Submission_Links.txt`.
3. Every member must appear and present technical content in the 20-minute video.
4. Keep the GitHub repository public and unmodified until marks are released.

Good luck!
