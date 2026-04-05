"""
clean_dataset.py
════════════════
Cleans a raw IDM-collected dataset before training.

Three categories of bad data found in the raw dataset:

  1. NUDGE CONTAMINATION (|steering| > 0.7)
     During recovery data collection, pressing A/D records the keypress
     action as a training label (steering=±1.0). These 3-frame spikes
     teach the model to randomly output full-lock steering.
     → Remove all rows with |steering| > 0.7

  2. COASTING FRAMES (throttle=0, brake=0, steer≈0, speed>5)
     IDM coasts on straights — throttle=0 for 36% of frames.
     The model learns "barely use throttle" causing sluggish driving.
     → Remove coasting frames (keep only frames where IDM is actively driving)

  3. PHYSICS-STUCK FRAMES (speed frozen at one value)
     During nudge recovery, the physics engine occasionally freezes
     the car speed while steering flips ±1.0.
     → Remove frames with identical speed across many consecutive rows

  4. STEERING NOISE (0 < |steering| < 0.01)
     IDM outputs floating-point noise like 3.7e-8, 0.007 on straight roads.
     The model learns these tiny values as "correct for straights" → drift.
     → Zero out steering values below 0.01 dead-zone

Run:
  python clean_dataset.py
  python train.py --epochs 40
"""

import pandas as pd
import numpy as np
import shutil
import os

CSV_PATH = "dataset/labels.csv"

if not os.path.exists(CSV_PATH):
    print(f"ERROR: {CSV_PATH} not found. Run from project root.")
    exit()

df = pd.read_csv(CSV_PATH)
original_len = len(df)
print(f"Original dataset : {original_len} rows")
print(f"Steering std     : {df['steering'].std():.5f}")
print(f"Throttle mean    : {df['throttle'].mean():.5f}")

# backup
backup = CSV_PATH.replace(".csv", "_backup.csv")
shutil.copy(CSV_PATH, backup)
print(f"\nBacked up to     : {backup}")

# ── Fix 1: Remove nudge-contaminated rows ─────────────────────────────────────
nudge_mask = df['steering'].abs() > 0.7
print(f"\n[Fix 1] Nudge rows (|steer|>0.7)       : {nudge_mask.sum():5d} → removed")
df = df[~nudge_mask].reset_index(drop=True)

# ── Fix 2: Remove physics-stuck frames ───────────────────────────────────────
# Detect speed values that repeat identically many times (physics freeze)
speed_counts = df['speed'].round(4).value_counts()
stuck_speeds = speed_counts[speed_counts > 50].index.tolist()
if stuck_speeds:
    stuck_mask = df['speed'].round(4).isin(stuck_speeds) & \
                 (df['steering'].abs() > 0.5)
    print(f"[Fix 2] Physics-stuck rows             : {stuck_mask.sum():5d} → removed")
    df = df[~stuck_mask].reset_index(drop=True)
else:
    print(f"[Fix 2] Physics-stuck rows             :     0 → none found")

# ── Fix 3: Remove coasting frames ─────────────────────────────────────────────
coast_mask = (df['throttle'] == 0.0) & \
             (df['brake'] == 0.0)    & \
             (df['steering'].abs() < 0.01) & \
             (df['speed'] > 5.0)
print(f"[Fix 3] Coasting frames (thr=brk=steer≈0): {coast_mask.sum():5d} → removed")
df = df[~coast_mask].reset_index(drop=True)

# ── Fix 4: Zero out steering noise ────────────────────────────────────────────
DEAD_ZONE = 0.01
noise_mask = (df['steering'].abs() > 0) & (df['steering'].abs() < DEAD_ZONE)
print(f"[Fix 4] Noise frames (0<|steer|<{DEAD_ZONE})  : {noise_mask.sum():5d} → zeroed")
df.loc[noise_mask, 'steering'] = 0.0
df.loc[df['prev_steering'].abs() < DEAD_ZONE, 'prev_steering'] = 0.0

# ── Summary ───────────────────────────────────────────────────────────────────
removed = original_len - len(df)
print(f"\nCleaned dataset  : {len(df)} rows  (removed {removed}, {100*removed/original_len:.0f}%)")
print(f"Steering std     : {df['steering'].std():.5f}  (should be similar to before)")
print(f"Throttle mean    : {df['throttle'].mean():.5f}  (should be higher than before)")
print(f"Throttle==0      : {(df['throttle']==0).sum()} ({100*(df['throttle']==0).mean():.0f}%)  (should be 0%)")
print(f"\nEvent distribution:")
print(df['event'].value_counts().to_string())

df.to_csv(CSV_PATH, index=False)
print(f"\nSaved to         : {CSV_PATH}")
print(f"Now run          : python train.py --epochs 40")
