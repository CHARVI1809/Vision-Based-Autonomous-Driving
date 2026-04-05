"""
dataset_collection_manual.py  –  MetaDrive 0.4.3
=================================================
Simple map: straight → right curve → straight → left curve → repeat
No roundabouts. No complex config. Just S and C blocks.

Left curve trick: Two right curves back-to-back create an S-bend.
Use multiple different seeds (NUM_MAPS=20) so the car sees varied roads.

COLLECTION_MODE:
  "autopilot" → Phase 1: IDM drives, you watch, ~8000 frames, 10-15 min
  "recovery"  → Phase 2: IDM drives, press A/D to drift car, ~3000 frames
"""

import os, csv, numpy as np, cv2
from collections import Counter

from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.policy.idm_policy import IDMPolicy

# ── change this between Phase 1 and Phase 2 ──────────────────────────────────
COLLECTION_MODE = "recovery"   # "autopilot"  or  "recovery"
# ─────────────────────────────────────────────────────────────────────────────

DATASET_DIR = "dataset"
IMAGES_DIR  = os.path.join(DATASET_DIR, "images")
CSV_PATH    = os.path.join(DATASET_DIR, "labels.csv")
CAMERA_W    = 128
CAMERA_H    = 128
MAX_STEPS   = 120_000
NUM_MAPS    = 20        # 20 different road seeds = varied curve directions
START_SEED  = 100
MIRROR_RARE = 2         # save 2 flipped copies of every curve frame
NUDGE_STEPS = 25        # how many steps to apply drift steering
NUDGE_STEER = 0.6       # how hard to steer during drift

# Simple map: alternating curves with short straights between them
# MetaDrive 0.4.3 only supports positive "angle" — no "direction" key
# Left vs right is determined by the random seed across 20 episodes
MAP_CFG = dict(
    type   = MapGenerateMethod.PG_MAP_FILE,
    config = [
        None,
        {"id": "S", "pre_block_socket_index": 0, "length": 50},
        {"id": "C", "pre_block_socket_index": 0, "length": 60, "angle": 60},
        {"id": "S", "pre_block_socket_index": 0, "length": 30},
        {"id": "C", "pre_block_socket_index": 0, "length": 60, "angle": 90},
        {"id": "S", "pre_block_socket_index": 0, "length": 30},
        {"id": "C", "pre_block_socket_index": 0, "length": 60, "angle": 60},
        {"id": "S", "pre_block_socket_index": 0, "length": 50},
    ],
    lane_num   = 3,
    lane_width = 3.5,
)
# Curve length vs straight: 180m curve, 160m straight → ~53% curve time


def get_action(info: dict):
    steer    = float(info.get("steering",     0.0))
    accel    = float(info.get("acceleration", 0.0))
    throttle = max(0.0,  accel)
    brake    = max(0.0, -accel)
    return steer, throttle, brake


def classify(info, steer, throttle, brake):
    if info.get("crash", False) or info.get("crash_vehicle", False):
        return "collision"
    if info.get("out_of_road", False):
        return "offroad"
    if abs(steer) > 0.20:   return "sharp_turn"
    if abs(steer) > 0.08:   return "gentle_turn"
    if brake > 0.30:        return "braking"
    if min(info.get("lateral_to_left", 999),
           info.get("lateral_to_right", 999)) < 1.2:
        return "lane_departure"
    return "normal"


def make_env():
    return MetaDriveEnv(dict(
        use_render              = True,
        manual_control          = False,
        traffic_density         = 0.0,
        window_size             = (1200, 800),
        map_config              = MAP_CFG,
        num_scenarios           = NUM_MAPS,
        start_seed              = START_SEED,
        random_lane_width       = False,
        random_lane_num         = False,
        random_spawn_lane_index = False,
        agent_configs           = {"default_agent": dict(
                                       spawn_lane_index=(">", ">>", 1))},
        image_observation       = True,
        sensors                 = {"rgb_camera": (RGBCamera, CAMERA_W, CAMERA_H)},
        vehicle_config          = dict(image_source="rgb_camera"),
        stack_size              = 1,
        norm_pixel              = True,
        horizon                 = 3000,
        out_of_road_done        = True,
        crash_vehicle_done      = False,
        agent_policy            = IDMPolicy,
    ))


def key_down(env, key):
    try:
        return env.engine.mouseWatcherNode.is_button_down(key)
    except Exception:
        try:
            return env.engine.is_pressing_key(key)
        except Exception:
            return False


def bgr(obs):
    return cv2.cvtColor((obs["image"][..., -1] * 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR)


def save(idx, img):
    p = f"frame_{idx:07d}.jpg"
    cv2.imwrite(os.path.join(IMAGES_DIR, p), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return os.path.join("images", p)


def show_stats(ev, n):
    total = sum(ev.values())
    print(f"\n  ── {n} frames ──────────────────────────────────────")
    for e, (lo, hi) in [("normal",(40,60)),("gentle_turn",(15,25)),
                         ("sharp_turn",(8,15)),("braking",(5,12)),
                         ("lane_departure",(5,15))]:
        cnt = ev.get(e, 0)
        pct = 100 * cnt / max(total, 1)
        bar = "█" * int(25 * cnt / max(total, 1))
        ok  = "✔" if lo <= pct <= hi else ("▲" if pct > hi else "▼")
        print(f"  {e:16s} {cnt:5d} ({pct:4.0f}%)  {bar}  {ok}")
    print()


def collect():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # handle existing data
    if os.path.exists(CSV_PATH):
        ans = input("  Existing dataset found. Delete? [y/N]: ")
        if ans.strip().lower() == "y":
            import shutil; shutil.rmtree(DATASET_DIR)
            os.makedirs(IMAGES_DIR, exist_ok=True)
            mode = "w"; frame_idx = 0
            print("  Deleted.\n")
        else:
            mode = "a"
            with open(CSV_PATH) as f:
                frame_idx = max(0, sum(1 for _ in f) - 1)
            print(f"  Appending from frame {frame_idx}\n")
    else:
        mode = "w"; frame_idx = 0

    fh     = open(CSV_PATH, mode, newline="")
    writer = csv.writer(fh)
    if mode == "w":
        writer.writerow(["image_path","speed","prev_steering",
                         "steering","throttle","brake","event"])

    env   = make_env()
    obs, info = env.reset()

    prev_steer = 0.0
    ep         = 1
    ev         = Counter()
    nleft      = 0      # nudge steps remaining
    ndir       = 0.0

    print("=" * 55)
    if COLLECTION_MODE == "autopilot":
        print("  AUTOPILOT — car drives itself")
        print("  Run 10-15 min. Press Q to stop.")
    else:
        print("  RECOVERY — IDM drives, you drift with A/D")
        print("  A = drift left   D = drift right")
        print("  Each press drifts for 25 steps then IDM corrects.")
        print("  Press Q to stop.")
    print("=" * 55)

    for step in range(MAX_STEPS):

        # recovery: apply nudge or let IDM drive
        if COLLECTION_MODE == "recovery":
            if nleft <= 0:
                if key_down(env, "a") or key_down(env, "arrow_left"):
                    nleft = NUDGE_STEPS; ndir = -1.0
                    print(f"  [{frame_idx}] ← drift left")
                elif key_down(env, "d") or key_down(env, "arrow_right"):
                    nleft = NUDGE_STEPS; ndir = +1.0
                    print(f"  [{frame_idx}] → drift right")
            if nleft > 0:
                obs, reward, term, trunc, info = env.step([NUDGE_STEER * ndir, 0.5])
                nleft -= 1
            else:
                obs, reward, term, trunc, info = env.step([0, 0])
        else:
            obs, reward, term, trunc, info = env.step([0, 0])

        steer, thr, brk = get_action(info)
        speed = float(getattr(env.agent, "speed", 0.0))

        # startup confirmation
        if step == 40:
            ok = thr > 0.01 or abs(steer) > 0.001
            print(f"\n  Step 40: steer={steer:.4f} thr={thr:.4f} speed={speed:.1f}")
            print(f"  {'✔ IDM working!' if ok else '⚠ still warming up...'}\n")

        event = classify(info, steer, thr, brk)

        if "image" not in obs:
            if term or trunc:
                obs, info = env.reset(); ep += 1; prev_steer = 0.0; nleft = 0
            continue

        img     = bgr(obs)
        path    = save(frame_idx, img)
        writer.writerow([path, f"{speed:.4f}", f"{prev_steer:.4f}",
                         f"{steer:.4f}", f"{thr:.4f}", f"{brk:.4f}", event])
        ev[event] += 1
        frame_idx += 1

        # mirror augmentation for turning frames
        if event in ("sharp_turn", "gentle_turn", "lane_departure") \
                and abs(steer) > 0.05:
            flipped = cv2.flip(img, 1)
            for _ in range(MIRROR_RARE if event == "sharp_turn" else 1):
                p2 = save(frame_idx, flipped)
                writer.writerow([p2, f"{speed:.4f}", f"{-prev_steer:.4f}",
                                  f"{-steer:.4f}", f"{thr:.4f}", f"{brk:.4f}", event])
                ev[event] += 1; frame_idx += 1

        prev_steer = steer

        if frame_idx % 1000 == 0:
            print(f"  ep={ep} frame={frame_idx} speed={speed:.1f} steer={steer:+.3f}")
            show_stats(ev, frame_idx)

        if term or trunc:
            print(f"  → ep {ep} done. frames={frame_idx}")
            obs, info = env.reset(); ep += 1; prev_steer = 0.0; nleft = 0

        try:
            if env.engine.is_pressing_key("q"):
                break
        except Exception:
            pass

    env.close(); fh.close()

    print(f"\n  Done. {frame_idx} frames saved.")
    show_stats(ev, frame_idx)
    import pandas as pd
    df = pd.read_csv(CSV_PATH)
    print(f"  steering std  = {df['steering'].std():.4f}  (want > 0.08)")
    print(f"  throttle mean = {df['throttle'].mean():.4f}  (want > 0.20)")
    if COLLECTION_MODE == "autopilot":
        print("\n  → Change COLLECTION_MODE = 'recovery' then run again (type N)")
    else:
        print("\n  → python fix_labels.py")
        print("  → python train.py")


if __name__ == "__main__":
    collect()
