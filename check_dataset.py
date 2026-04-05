"""
check_dataset.py
Run this BEFORE cleaning to understand your data properly.
"""
import pandas as pd
import numpy as np

CSV_PATH = "dataset/labels.csv"
df = pd.read_csv(CSV_PATH)

s = df['steering'].abs()

print(f"Total frames: {len(df)}")
print(f"\nExact breakdown of |steering| values:")
print(f"  == 0.0000 exactly  : {(s == 0.0).sum():5d} ({100*(s==0.0).mean():.1f}%)")
print(f"  0.0000 - 0.0100    : {((s > 0) & (s <= 0.01)).sum():5d}")
print(f"  0.0100 - 0.0200    : {((s > 0.01) & (s <= 0.02)).sum():5d}")
print(f"  0.0200 - 0.0400    : {((s > 0.02) & (s <= 0.04)).sum():5d}")
print(f"  0.0400 - 0.0600    : {((s > 0.04) & (s <= 0.06)).sum():5d}")
print(f"  0.0600 - 0.0800    : {((s > 0.06) & (s <= 0.08)).sum():5d}")
print(f"  0.0800 - 0.1200    : {((s > 0.08) & (s <= 0.12)).sum():5d}")
print(f"  0.1200 - 0.2000    : {((s > 0.12) & (s <= 0.20)).sum():5d}")
print(f"  > 0.2000           : {(s > 0.20).sum():5d}")

print(f"\nAlready-zero frames   : {(s == 0.0).sum()}")
print(f"Near-zero (< 0.02)   : {(s < 0.02).sum()}")
print(f"This includes already-zero: {(s == 0.0).sum()}")
print(f"Genuinely tiny (0.0 < s < 0.02): {((s > 0) & (s < 0.02)).sum()}")

# Check if this is the cleaned version or backup
print(f"\nIs this already cleaned? (many exact zeros = yes)")
exact_zeros_pct = 100 * (s == 0.0).mean()
if exact_zeros_pct > 30:
    print(f"  ⚠ {exact_zeros_pct:.0f}% exact zeros — this looks like the CLEANED version")
    print(f"  Restore backup first: copy dataset\\labels_backup.csv dataset\\labels.csv")
else:
    print(f"  ✔ {exact_zeros_pct:.0f}% exact zeros — this looks like the original")
