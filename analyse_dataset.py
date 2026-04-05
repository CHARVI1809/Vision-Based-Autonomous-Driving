"""
analyse_dataset.py  (v2 - deeper analysis)
"""
import pandas as pd
import numpy as np
import os

CSV_PATH = "dataset/labels.csv"
df = pd.read_csv(CSV_PATH)
print(f"Total frames: {len(df)}")
print(f"\nEvent distribution:")
print(df['event'].value_counts().to_string())

s     = df['steering'].values
prev  = df['prev_steering'].values

# The real curve-exit problem: 
# Look at what happens in the 10 frames AFTER steering drops below 0.08
# These are the frames the model needs to output near-zero steering for
print(f"\nSteering decay analysis (what happens right after a curve):")
buckets = {'0.00-0.02': 0, '0.02-0.05': 0, '0.05-0.08': 0, '0.08-0.15': 0, '>0.15': 0}
post_curve_steers = []
for i in range(1, len(s)):
    if abs(prev[i]) > 0.15 and abs(s[i]) < abs(prev[i]):
        post_curve_steers.append(abs(s[i]))

if post_curve_steers:
    for v in post_curve_steers:
        if v < 0.02:   buckets['0.00-0.02'] += 1
        elif v < 0.05: buckets['0.02-0.05'] += 1
        elif v < 0.08: buckets['0.05-0.08'] += 1
        elif v < 0.15: buckets['0.08-0.15'] += 1
        else:          buckets['>0.15']     += 1
    total_pc = len(post_curve_steers)
    print(f"  Frames after peak steering ({total_pc} total):")
    for k, v in buckets.items():
        bar = '█' * int(30 * v / max(total_pc, 1))
        print(f"    steer {k:10s}: {v:4d} ({100*v/total_pc:4.0f}%)  {bar}")
    
    near_zero = buckets['0.00-0.02'] + buckets['0.02-0.05']
    print(f"\n  Frames decaying to near-zero (<0.05): {near_zero} ({100*near_zero/total_pc:.0f}%)")
    if near_zero / total_pc < 0.20:
        print(f"  ⚠ IDM steering decays slowly — most post-curve frames still have")
        print(f"    significant steering. The model learns 'keep steering after curves'.")
        print(f"\n  THIS IS THE ROOT CAUSE.")
        print(f"  The IDM doesn't snap to zero — it decays gradually.")
        print(f"  So frames labeled 'normal' after a curve still have steering=0.05-0.15.")
        print(f"  The model learns those non-zero values as correct for straight roads.")

# Show actual steering values right after peak
print(f"\nSteering value distribution across ALL frames:")
bins = [0, 0.02, 0.05, 0.08, 0.12, 0.20, 0.30, 1.01]
lbls = ['0-0.02','0.02-0.05','0.05-0.08','0.08-0.12','0.12-0.20','0.20-0.30','>0.30']
abs_s = np.abs(s)
for i, lbl in enumerate(lbls):
    cnt = ((abs_s >= bins[i]) & (abs_s < bins[i+1])).sum()
    pct = 100 * cnt / len(s)
    bar = '█' * int(30 * cnt / len(s))
    print(f"  {lbl:12s}: {cnt:5d} ({pct:4.1f}%)  {bar}")

print(f"\nKey insight:")
print(f"  Frames with |steer| in 0.05-0.12 range: "
      f"{((abs_s >= 0.05) & (abs_s < 0.12)).sum()}")
print(f"  These are 'ambiguous' frames — on a straight but still slightly steering.")
print(f"  The model learns this as normal straight-road behaviour.")
