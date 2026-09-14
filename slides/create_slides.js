const PptxGenJS = require("pptxgenjs");
const pptxgen = PptxGenJS;

let pres = new pptxgen();
pres.defineLayout({ name: "WIDE", width: 13.333, height: 7.5 });
pres.layout = "WIDE";
pres.author = "ADDO, Andrews, George";
pres.title = "PPO-1 Autonomous Vehicle Parking";

// Colour scheme
const BLUE = "1B4F72";
const LIGHT = "D4E6F1";
const DARK = "1C2833";
const ACCENT = "2874A6";

function addTitleBar(slide, title) {
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 13.333, h: 1.0, fill: { color: BLUE } });
  slide.addText(title, { x: 0.4, y: 0.25, w: 12.5, h: 0.5, fontSize: 24, color: "FFFFFF", bold: true, fontFace: "Arial" });
}

// ========== SLIDE 1: Title ==========
let s1 = pres.addSlide();
s1.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 13.333, h: 7.5, fill: { color: BLUE } });
s1.addText("Autonomous Vehicle Parking", { x: 0.5, y: 2.0, w: 12.3, h: 0.8, fontSize: 36, color: "FFFFFF", bold: true, align: "center", fontFace: "Arial" });
s1.addText("using Proximal Policy Optimisation (PPO)", { x: 0.5, y: 2.8, w: 12.3, h: 0.5, fontSize: 22, color: LIGHT, align: "center", fontFace: "Arial" });
s1.addText("DSCD 614 – Reinforcement Learning  |  University of Ghana", { x: 0.5, y: 4.0, w: 12.3, h: 0.4, fontSize: 16, color: "FFFFFF", align: "center", fontFace: "Arial" });
s1.addText("ADDO, Austine Gamey (22424506)\nAndrews Anseiku Junior (22427819)\nGeorge Manuel (22424752)", { x: 0.5, y: 5.0, w: 12.3, h: 1.2, fontSize: 16, color: "FFFFFF", align: "center", fontFace: "Arial" });

// ========== SLIDE 2: Problem & Aims ==========
let s2 = pres.addSlide();
addTitleBar(s2, "1. Problem & Aims");
s2.addText("Problem", { x: 0.5, y: 1.3, w: 5.5, h: 0.4, fontSize: 18, bold: true, color: BLUE, fontFace: "Arial" });
s2.addText("Navigate a vehicle from an arbitrary initial pose into a designated parking space while avoiding collisions and achieving correct final orientation.", { x: 0.5, y: 1.8, w: 5.8, h: 1.5, fontSize: 15, color: DARK, fontFace: "Arial" });
s2.addText("Aims", { x: 7.0, y: 1.3, w: 5.5, h: 0.4, fontSize: 18, bold: true, color: BLUE, fontFace: "Arial" });
s2.addText([
  { text: "Formulate the task as an MDP", options: { breakLine: true } },
  { text: "Train a PPO agent (3 seeds)", options: { breakLine: true } },
  { text: "Compare against a rule-based baseline", options: { breakLine: true } },
  { text: "Analyse convergence & limitations", options: { breakLine: true } }
], { x: 7.0, y: 1.8, w: 5.5, h: 2.0, fontSize: 15, color: DARK, fontFace: "Arial" });
s2.addText("Environment: HighwayEnv parking-v0  |  Algorithm: PPO (Stable-Baselines3)", { x: 0.5, y: 6.5, w: 12.3, h: 0.4, fontSize: 14, color: ACCENT, fontFace: "Arial" });

// ========== SLIDE 3: MDP Formulation ==========
let s3 = pres.addSlide();
addTitleBar(s3, "2. Markov Decision Process Formulation");
s3.addText("State (12-D)", { x: 0.4, y: 1.2, w: 4.0, h: 0.35, fontSize: 16, bold: true, color: BLUE, fontFace: "Arial" });
s3.addText("Current kinematics (x,y,vx,vy,cos h,sin h)\n+ Desired goal (same 6 features)", { x: 0.4, y: 1.55, w: 4.0, h: 0.9, fontSize: 13, color: DARK, fontFace: "Arial" });

s3.addText("Action (2-D continuous)", { x: 4.7, y: 1.2, w: 4.0, h: 0.35, fontSize: 16, bold: true, color: BLUE, fontFace: "Arial" });
s3.addText("Steering δ ∈ [-1,1]\nAcceleration a ∈ [-1,1]", { x: 4.7, y: 1.55, w: 4.0, h: 0.9, fontSize: 13, color: DARK, fontFace: "Arial" });

s3.addText("Discount γ = 0.99", { x: 9.0, y: 1.2, w: 3.8, h: 0.35, fontSize: 16, bold: true, color: BLUE, fontFace: "Arial" });
s3.addText("Matches short parking horizon\n(~100 effective steps)", { x: 9.0, y: 1.55, w: 3.8, h: 0.9, fontSize: 13, color: DARK, fontFace: "Arial" });

s3.addText("Reward (native HighwayEnv)", { x: 0.4, y: 2.8, w: 12.5, h: 0.35, fontSize: 16, bold: true, color: BLUE, fontFace: "Arial" });
s3.addText("r_t = −∥s_t − s_g∥_{W,p}^p  +  c · 𝟙_collision     (W = [1.0, 0.3, 0, 0, 0.02, 0.02], p = 0.5, c = −5)", { x: 0.4, y: 3.2, w: 12.5, h: 0.5, fontSize: 14, color: DARK, fontFace: "Arial" });

s3.addText("Termination: collision OR goal reached     |     Truncation: 100 environment steps", { x: 0.4, y: 4.0, w: 12.5, h: 0.4, fontSize: 14, color: DARK, fontFace: "Arial" });
s3.addText("Markov property: holds under the kinematic bicycle model (full state is observed).", { x: 0.4, y: 4.6, w: 12.5, h: 0.4, fontSize: 14, color: DARK, fontFace: "Arial" });

// ========== SLIDE 4: Methodology ==========
let s4 = pres.addSlide();
addTitleBar(s4, "3. Methodology");
s4.addText("PPO Agent", { x: 0.4, y: 1.3, w: 6.0, h: 0.4, fontSize: 18, bold: true, color: BLUE, fontFace: "Arial" });
s4.addText("• Stable-Baselines3 PPO\n• MLP [256, 256] actor & critic\n• 3 independent seeds (42, 123, 456)\n• Hyperparameters held constant\n• 300 000 timesteps per seed", { x: 0.4, y: 1.8, w: 6.0, h: 2.5, fontSize: 15, color: DARK, fontFace: "Arial" });

s4.addText("Rule-based Baseline", { x: 7.0, y: 1.3, w: 5.8, h: 0.4, fontSize: 18, bold: true, color: BLUE, fontFace: "Arial" });
s4.addText("• Geometric heading controller\n• Approach phase → alignment phase\n• Deterministic\n• Evaluated under identical protocol", { x: 7.0, y: 1.8, w: 5.8, h: 2.2, fontSize: 15, color: DARK, fontFace: "Arial" });

s4.addText("Evaluation: ≥30 episodes/seed, deterministic policy, held-out seeds, mean ± std reported", { x: 0.4, y: 5.0, w: 12.5, h: 0.5, fontSize: 14, color: ACCENT, fontFace: "Arial" });

// ========== SLIDE 5: Results placeholder ==========
let s5 = pres.addSlide();
addTitleBar(s5, "4. Results (to be completed after full training)");
s5.addText("Training curves (mean ± std across 3 seeds) and quantitative comparison tables will appear here after the full 300 k-step training runs finish.", { x: 0.5, y: 2.5, w: 12.3, h: 1.5, fontSize: 18, color: DARK, align: "center", fontFace: "Arial" });
s5.addText("All figures are regenerated from committed raw logs.", { x: 0.5, y: 4.5, w: 12.3, h: 0.5, fontSize: 15, color: ACCENT, align: "center", fontFace: "Arial" });

// ========== SLIDE 6: Limitations ==========
let s6 = pres.addSlide();
addTitleBar(s6, "5. Principal Limitations");
s6.addText([
  { text: "Simplified kinematic bicycle model (no tyre slip, actuator delay)", options: { breakLine: true } },
  { text: "Fully observable, noise-free state (real systems need estimation)", options: { breakLine: true } },
  { text: "Static empty parking bay (no pedestrians or moving vehicles)", options: { breakLine: true } },
  { text: "Limited compute within a 14-day student project window", options: { breakLine: true } },
  { text: "Deterministic evaluation policy; deployment needs safety layers", options: { breakLine: true } }
], { x: 0.8, y: 1.8, w: 11.5, h: 3.5, fontSize: 18, color: DARK, fontFace: "Arial" });

// ========== SLIDE 7: Conclusion ==========
let s7 = pres.addSlide();
addTitleBar(s7, "6. Conclusion & Further Work");
s7.addText("We formulated autonomous parking as a continuous-control MDP, implemented a PPO agent and a rule-based baseline, and defined a reproducible multi-seed protocol that meets all examination requirements.", { x: 0.5, y: 1.5, w: 12.3, h: 1.5, fontSize: 16, color: DARK, fontFace: "Arial" });
s7.addText("Further work: parked vehicles as obstacles, domain randomisation, transfer to higher-fidelity simulators (CARLA).", { x: 0.5, y: 3.3, w: 12.3, h: 1.0, fontSize: 16, color: DARK, fontFace: "Arial" });
s7.addText("Thank you – Questions?", { x: 0.5, y: 5.5, w: 12.3, h: 0.6, fontSize: 22, bold: true, color: BLUE, align: "center", fontFace: "Arial" });

pres.writeFile({ fileName: "/home/workdir/artifacts/PPO1_AUTONOMOUS_PARKING_FINAL/slides/PPO1_Demonstration_Slides.pptx" })
  .then(() => console.log("Slides created successfully"))
  .catch(err => console.error(err));
