# Vision-Based Autonomous Driving 

An end-to-end autonomous driving project built with [MetaDrive 0.4.3](https://github.com/metadriverse/metadrive) and PyTorch. A CNN learns to drive by imitating an IDM autopilot — no hand-coded rules, no lidar, just a front camera.

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange) ![MetaDrive](https://img.shields.io/badge/MetaDrive-0.4.3-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

**Result: 3/3 episodes completed, 0% off-road rate, 0% crash rate.**

---

## What It Does

The car sees only its front camera (128×128 RGB) and outputs steering, throttle and brake in real time. No map, no lidar, no privileged information.

```
Camera (128×128) ──► CNN ──► [steering, throttle, brake]
speed + prev_steer ──► MLP ──┘
```

---

## Project Structure

```
Vision-Based-Autonomous-Driving/
│
├── dataset_collection_manual.py   # IDM autopilot + recovery data collection
├── clean_dataset.py               # Removes nudge contamination and coasting frames
├── fix_labels.py                  # Re-labels CSV with correct event thresholds
├── analyse_dataset.py             # Dataset quality diagnostics
├── check_dataset.py               # Steering value distribution check
├── debug_idm.py                   # Verifies IDM action reading works
├── train.py                       # Model definition + training pipeline
│
├── autonomous_drive/
│   └── autonomous_drive.py        # Autonomous inference with lane monitoring
│
├── dataset/                       # gitignored — created at runtime
│   ├── images/                    # 128×128 JPEG frames
│   └── labels.csv
│
├── checkpoints/                   # gitignored — created at runtime
│   ├── best_model.pth
│   └── last_model.pth
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Five-Phase Development

```
Phase 1 · Setup           → install, verify MetaDrive, run debug_idm.py
Phase 2 · Data Collection → IDM autopilot (8k frames) + recovery (3k frames)
                            + dataset cleaning (remove nudge contamination)
Phase 3 · Model           → DrivingCNN — lightweight 4-stage CNN, ~262k params
Phase 4 · Training        → 40 epochs, ~60-100 min on CPU
Phase 5 · Autonomous      → 3/3 episodes completed, lane change monitoring
```

---

## Quick Start

### 1 · Install

```bash
pip install metadrive-simulator==0.4.3
pip install -r requirements.txt
```

### 2 · Verify environment

```bash
python debug_idm.py
```

Confirm `info['steering']` and `info['acceleration']` are non-zero by step 30.

### 3 · Collect Phase 1 — autopilot

```python
# dataset_collection_manual.py, line ~30:
COLLECTION_MODE = "autopilot"
```
```bash
python dataset_collection_manual.py   # type y to start fresh
```
Run 10–15 minutes (~8,000 frames). Press Q to stop.

### 4 · Collect Phase 2 — recovery

```python
COLLECTION_MODE = "recovery"
```
```bash
python dataset_collection_manual.py   # type N to keep Phase 1 data
```
Press `A` or `D` to drift car toward edge, then release. IDM corrects back — that correction is recovery data. Run 5–7 minutes (~3,000 frames).

### 5 · Clean the dataset

```bash
python clean_dataset.py
```

This removes four types of bad data (see Dataset Cleaning section below).

### 6 · Verify

```bash
python fix_labels.py
python analyse_dataset.py
```

### 7 · Train

```bash
python train.py --epochs 40
```

~60–100 minutes on CPU. Target: `steer_MAE < 0.05` by epoch 30.

### 8 · Run autonomous

```bash
python autonomous_drive/autonomous_drive.py --episodes 3
```

---

## Dataset Cleaning

The raw IDM dataset contains several categories of bad data that must be removed before training. Run `python clean_dataset.py` to fix all of them automatically.

### Fix 1 — Nudge contamination

During recovery data collection, pressing A/D records the keypress action as a training label:
```
frame N:   steering = 0.00   (normal driving)
frame N+1: steering = +1.00  ← A/D keypress recorded as label
frame N+2: steering = -1.00  ← physics bounce
frame N+3: steering = -1.00
frame N+4: steering = -0.19  ← IDM takes back over
```
These spikes teach the model to randomly output full-lock steering. **Remove all rows with `|steering| > 0.7`.**

### Fix 2 — Physics-stuck frames

During nudge recovery the physics engine occasionally freezes car speed while steering flips ±1.0. Detected as speed repeating identically many times with large steering. **Remove these rows.**

### Fix 3 — Coasting frames

IDM coasts on long straights with `throttle=0`. This was 36% of the raw dataset. The model learns "barely use throttle" causing sluggish driving that can't handle curves. **Remove frames where throttle=0, brake=0, steer≈0, speed>5 m/s.**

### Fix 4 — Steering noise dead-zone

IDM outputs floating-point noise (`0.0000003`, `0.007`) on perfectly straight roads. The model learns these tiny values as "correct for straights" then outputs them at inference, causing slow drift. **Zero out any `|steering| < 0.01`.**

**Result:** Raw dataset 11610 rows → cleaned 7989 rows. Throttle mean 0.08 → 0.12. Zero throttle 36% → 0%.

---

## Model Architecture

```
Input
  image   (3, 128, 128)
  scalars (2,)  [speed/30, prev_steering]

CNN Encoder
  Conv(3→32)   + BN + ReLU + MaxPool  →  64×64
  Conv(32→64)  + BN + ReLU + MaxPool  →  32×32
  Conv(64→128) + BN + ReLU + MaxPool  →  16×16
  Conv(128→128)+ BN + ReLU + MaxPool  →   8×8
  GlobalAvgPool                        →  (128,)

Scalar MLP
  Linear(2→32) + ReLU                 →  (32,)

Fusion
  concat(160) → Linear(160→128) → ReLU → Dropout(0.2)

Output Heads  (separate — no gradient interference)
  steer_head    → tanh    → steering ∈ [-1,  1]
  throttle_head → sigmoid → throttle ∈ [ 0,  1]
  brake_head    → sigmoid → brake    ∈ [ 0,  1]

Total parameters: ~262,275
```

Key design decisions:
- **GlobalAveragePool** — resolution-agnostic, fast to train
- **Separate output heads** — steering gradient does not interfere with throttle/brake
- **Zero-init steering weights** — prevents constant-bias bug on imbalanced datasets
- **Steering warmup** — first 5 epochs train steering loss only, then all outputs
- **No ImageNet normalisation** — images are raw `[0,1]` from MetaDrive `norm_pixel=True`

---

## Training Details

| Parameter | Value | Notes |
|---|---|---|
| Epochs | 40 | Model converges fully, no early stopping needed |
| Batch size | 16 | |
| Learning rate | 3e-4 | Cosine decay to 1e-6 |
| Steering loss weight | 5.0 | |
| Warmup epochs | 5 | Steering-only loss phase |
| Grad clip | 0.5 | |
| Oversampling | On | Rare events upweighted 3–5× |

Training results on cleaned 7989-frame dataset:
```
epoch 40: train_loss=0.0011  val_loss=0.0039  steer_MAE=0.020
```

---

## Autonomous Driving

```bash
python autonomous_drive/autonomous_drive.py [options]
```

| Parameter | Default | Description |
|---|---|---|
| `--max_speed` | 8.0 | Speed cap in m/s — must match training data max speed |
| `--max_thr` | 0.35 | Throttle cap |
| `--smooth` | 0.1 | EMA alpha for steering smoothing |
| `--centre_pull` | 0.5 | Extra decay when model says "go straight" |
| `--curve_exit_thresh` | 0.15 | Snap to zero threshold after curves |
| `--steer_scale` | 1.2 | Multiply raw steering output |
| `--episodes` | 3 | Number of episodes |

**Important:** `--max_speed` must match your training data's max speed. The model was trained on data where the car never exceeded 8.33 m/s. Running faster puts the model in unseen territory causing unreliable steering.

### Lane Change Detection

The autonomous script monitors the current lane index every step:
- Spawns in **middle lane (index 1)** always
- Prints a warning whenever the car moves to lane 0 (left) or lane 2 (right)
- Reports total lane changes per episode in the summary

Example output:
```
  ⚠ LANE CHANGE step= 312  lane 1 → 0  (LEFT)  speed=7.8  steer=-0.231
Episode 1 summary:
  Result       : ARRIVED ✔
  Lane changes : 1  at steps [312]
```

### Steering Post-Processing

Raw model output is filtered through three stages before being sent to the car:

```
Stage 1 — EMA:         smooth = 0.1 * smooth + 0.9 * raw
Stage 2 — Centre pull: if model says straight but smooth still high → decay by 0.5×
Stage 3 — Curve exit:  if smooth > 0.15 but raw < 0.05 → snap to 0.3× immediately
Dead-zone:             if |smooth| < 0.02 → set to 0.0
```

Stage 3 fixes the "car doesn't straighten after curves" problem caused by low curve-exit frame count in the dataset.

---

## Map Layout

```
start → straight(50m) → curve(60°) → straight(30m) → curve(90°)
      → straight(30m) → curve(60°) → straight(50m) → end
```

3 lanes · 3.5m width · car always spawns in middle lane (index 1).

Mirror augmentation saves a flipped copy of every curve frame with negated steering — giving the model both left and right turn data.

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| All CSV values zero | IDM action read from wrong attribute | Check `debug_idm.py` — use `info['steering']` |
| 70%+ normal frames | Map has too many straights | Shorten straight lengths in MAP_CFG |
| Car doesn't straighten after curve | Low curve-exit frames in dataset | Run `collect_curve_exits.py`, use `--curve_exit_thresh 0.10` |
| Car barely moves / throttle=0 | Coasting frames in dataset | Run `clean_dataset.py` to remove them |
| Random full-lock steering spikes | Nudge contamination in dataset | Run `clean_dataset.py` to remove |steer|>0.7 rows |
| Goes off-road at high speed | Speed outside training distribution | Set `--max_speed` to match training data max speed (8.0) |
| `AssertionError: Angle should be > 0` | Negative angle in curve config | MetaDrive 0.4.3 doesn't support negative angles |
| `KeyError: enable_lane_change` | Invalid vehicle_config key | Use `agent_configs` with `spawn_lane_index` |

---

## Requirements

```
metadrive-simulator==0.4.3
torch>=2.0
torchvision
numpy
opencv-python
pandas
pillow
```

---

## License

MIT
