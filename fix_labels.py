"""
fix_labels.py
─────────────
Re-labels an existing dataset CSV with corrected event thresholds.
Run this on your existing dataset BEFORE retraining — no need to re-record.

Also prints a full analysis of your steering patterns so you can
understand what your keyboard actually produces.

Usage
─────
  python fix_labels.py
  python fix_labels.py --csv dataset/labels.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="dataset/labels.csv")
    p.add_argument("--out", type=str, default=None,
                   help="Output path. Defaults to overwriting input CSV.")
    return p.parse_args()


def classify_event_v3(row) -> str:
    """
    Corrected classifier — same logic as dataset_collection_manual.py v3.
    Applied row-by-row to fix an existing mislabelled CSV.
    """
    steer    = abs(float(row["steering"]))
    brake    = float(row["brake"])
    throttle = float(row["throttle"])

    # MetaDrive flags — these columns won't exist in old CSVs,
    # so we fall back gracefully
    crash    = bool(row.get("crash",        False))
    offroad  = bool(row.get("out_of_road",  False))

    if crash:                   return "collision"
    if offroad:                 return "offroad"
    if steer > 0.20:            return "sharp_turn"
    if steer > 0.08:            return "gentle_turn"
    if brake > 0.30:            return "braking"
    return "normal"
    # Note: lane_departure can't be recovered from the CSV
    # because lateral_to_left/right weren't saved.
    # It will be missing from the re-labelled dataset —
    # that's fine; re-record with v9 script to get it back.


def analyse(df: pd.DataFrame, label: str):
    steer = df["steering"].abs()
    print(f"\n  ── {label} ({'N='+str(len(df))} frames) ──────────────────────")
    print(f"  Steering  mean={steer.mean():.4f}  std={steer.std():.4f}  "
          f"max={steer.max():.4f}  min={steer.min():.4f}")

    # histogram of absolute steering in buckets
    buckets = [0, 0.02, 0.05, 0.08, 0.12, 0.20, 0.30, 0.40, 0.50, 1.01]
    labels  = ["0–0.02 (noise)", "0.02–0.05", "0.05–0.08",
               "0.08–0.12 (gentle)", "0.12–0.20 (gentle)",
               "0.20–0.30 (sharp)", "0.30–0.40 (sharp)",
               "0.40–0.50 (very sharp)", "0.50+ (max)"]
    print(f"\n  Steering magnitude distribution:")
    for i in range(len(labels)):
        lo, hi  = buckets[i], buckets[i+1]
        mask    = (steer >= lo) & (steer < hi)
        cnt     = mask.sum()
        pct     = 100 * cnt / max(len(df), 1)
        bar     = "█" * min(40, int(40 * cnt / max(len(df), 1)))
        print(f"    {labels[i]:28s} {cnt:5d} ({pct:5.1f}%)  {bar}")

    # tap pattern detection
    # if many consecutive frames have steer→0 transitions, it's tap driving
    steers      = df["steering"].values
    tap_count   = 0
    for i in range(1, len(steers)):
        if abs(steers[i-1]) > 0.08 and abs(steers[i]) < 0.02:
            tap_count += 1
    tap_rate = 100 * tap_count / max(len(df), 1)
    print(f"\n  Tap-and-release pattern detected: {tap_count} times ({tap_rate:.1f}%)")
    if tap_rate > 5:
        print(f"  ⚠  HIGH TAP RATE — you are pressing/releasing A/D rapidly.")
        print(f"     HOLD the key through the entire curve instead.")
    else:
        print(f"  ✔  Tap rate OK — steering looks smooth.")

    # throttle analysis
    thr = df["throttle"]
    zero_thr = 100 * (thr == 0).sum() / max(len(df), 1)
    print(f"\n  Throttle=0 frames: {zero_thr:.1f}%", end="")
    if zero_thr > 60:
        print(f"  ⚠  Hold W while steering — don't coast through curves.")
    else:
        print(f"  ✔")

    print()


def main():
    args = parse_args()
    csv_path = Path(args.csv)

    if not csv_path.exists():
        print(f"  ERROR: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    print("=" * 65)
    print("  Dataset Label Fixer & Analyser")
    print(f"  Input : {csv_path}  ({len(df)} rows)")
    print("=" * 65)

    # ── analysis BEFORE fixing ────────────────────────────────────────────────
    print("\n  BEFORE fixing:")
    print(f"  Event distribution:\n{df['event'].value_counts().to_string()}")
    analyse(df, "BEFORE")

    # ── re-label ──────────────────────────────────────────────────────────────
    df["event_old"] = df["event"]
    df["event"]     = df.apply(classify_event_v3, axis=1)

    changed = (df["event"] != df["event_old"]).sum()
    print(f"\n  Re-labelled {changed} rows ({100*changed/len(df):.1f}%)")

    # ── analysis AFTER fixing ─────────────────────────────────────────────────
    print("\n  AFTER fixing:")
    print(f"  Event distribution:\n{df['event'].value_counts().to_string()}")
    analyse(df, "AFTER")

    # ── show transition table ─────────────────────────────────────────────────
    print("\n  Label change breakdown:")
    transition = df.groupby(["event_old", "event"]).size().reset_index(name="count")
    for _, row in transition.iterrows():
        arrow = "→"
        if row["event_old"] != row["event"]:
            print(f"    {row['event_old']:16s} {arrow} {row['event']:16s} : {row['count']}")

    # ── quality warnings ──────────────────────────────────────────────────────
    ev_counts = df["event"].value_counts()
    total     = len(df)
    print("\n  ── Quality Warnings ─────────────────────────────────────────")

    pct_normal = 100 * ev_counts.get("normal", 0) / total
    if pct_normal > 70:
        print(f"  ⚠  {pct_normal:.0f}% normal — too many straight frames.")
        print(f"     Re-record with more curve and recovery driving.")

    pct_turns = 100 * (ev_counts.get("sharp_turn", 0) +
                       ev_counts.get("gentle_turn", 0)) / total
    if pct_turns < 10:
        print(f"  ⚠  Only {pct_turns:.1f}% turning frames.")
        print(f"     You need at least 10%. Drive more curves.")

    steer_std = df["steering"].std()
    if steer_std < 0.08:
        print(f"  ⚠  Steering std={steer_std:.4f} is very low.")
        print(f"     This means almost no real turning in your data.")

    steer_mean = df["steering"].mean()
    if abs(steer_mean) > 0.05:
        print(f"  ⚠  Steering mean={steer_mean:.4f} — dataset is biased "
              f"{'right' if steer_mean > 0 else 'left'}.")
        print(f"     Drive equal amounts turning left and right.")

    if pct_normal <= 70 and pct_turns >= 10 and steer_std >= 0.08:
        print(f"  ✔  Dataset quality looks OK for training.")

    print()

    # ── save ──────────────────────────────────────────────────────────────────
    df = df.drop(columns=["event_old"])
    out_path = Path(args.out) if args.out else csv_path
    df.to_csv(out_path, index=False)
    print(f"  Saved fixed CSV → {out_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
