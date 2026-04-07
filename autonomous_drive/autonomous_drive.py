"""
autonomous_drive/autonomous_drive.py  –  MetaDrive 0.4.3
=========================================================
Added: lateral lane-keeping correction.

When the car drifts toward a lane edge (detected via lateral_to_left /
lateral_to_right from the info dict), a small corrective steering force
is added on top of the model output. This prevents the brief lane changes
seen during curves where centrifugal force pushes the car across the line.

Correction formula:
  If car is within LANE_KEEP_ZONE metres of left edge  → add +correction (steer right)
  If car is within LANE_KEEP_ZONE metres of right edge → add -correction (steer left)
  Correction magnitude scales linearly with proximity to edge.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse, time
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.map.pg_map import MapGenerateMethod

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import DrivingCNN

CAMERA_W = 128
CAMERA_H = 128

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

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def make_env(seed):
    return MetaDriveEnv(dict(
        use_render              = True,
        manual_control          = False,
        traffic_density         = 0.0,
        window_size             = (1200, 800),
        map_config              = MAP_CFG,
        num_scenarios           = 20,
        start_seed              = seed,
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
    ))


def get_lane_index(env) -> int:
    try:
        return int(env.agent.lane_index[2])
    except Exception:
        return -1


def lane_keep_correction(info: dict, zone: float, strength: float) -> float:
    """
    Computes a corrective steering value based on lateral distance to lane edges.

    zone     : distance from edge (metres) at which correction starts
    strength : max correction magnitude (added to model steering)

    Returns a value in [-strength, +strength]:
      positive = steer right (car drifted left)
      negative = steer left  (car drifted right)
    """
    lat_l = float(info.get("lateral_to_left",  99.0))
    lat_r = float(info.get("lateral_to_right", 99.0))

    correction = 0.0

    if lat_l < zone:
        # Too close to left edge — steer right
        # Scale: at edge (lat_l=0) → full strength; at zone boundary → 0
        t = 1.0 - (lat_l / zone)          # 0 at zone edge, 1 at lane edge
        correction = +strength * t

    elif lat_r < zone:
        # Too close to right edge — steer left
        t = 1.0 - (lat_r / zone)
        correction = -strength * t

    return correction


@torch.no_grad()
def get_raw(model, obs, speed, prev_steer, device):
    rgb_f = obs["image"][..., -1]
    img   = Image.fromarray((rgb_f * 255).astype(np.uint8))
    img_t = transform(img).unsqueeze(0).to(device)
    sca_t = torch.tensor([[speed / 30.0, prev_steer]],
                          dtype=torch.float32).to(device)
    out = model(img_t, sca_t).squeeze(0).cpu().numpy()
    return float(out[0]), float(out[1]), float(out[2])


def process_steering(raw, smooth, alpha, centre_pull, curve_exit_thresh):
    new_smooth = alpha * smooth + (1.0 - alpha) * raw
    if abs(raw) < 0.05 and abs(new_smooth) > 0.08:
        new_smooth *= centre_pull
    if abs(smooth) > curve_exit_thresh and abs(raw) < 0.05:
        new_smooth *= 0.3
    if abs(new_smooth) < 0.02:
        new_smooth = 0.0
    return new_smooth


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",        default="checkpoints/best_model.pth")
    p.add_argument("--episodes",          type=int,   default=3)
    p.add_argument("--seed",              type=int,   default=100)
    p.add_argument("--smooth",            type=float, default=0.1)
    p.add_argument("--centre_pull",       type=float, default=0.5)
    p.add_argument("--curve_exit_thresh", type=float, default=0.15)
    p.add_argument("--max_speed",         type=float, default=8.0)
    p.add_argument("--max_thr",           type=float, default=0.35)
    p.add_argument("--steer_scale",       type=float, default=1.2)
    # Lane keeping parameters
    p.add_argument("--lane_keep_zone",    type=float, default=1.2,
                   help="Distance from lane edge (m) at which correction starts")
    p.add_argument("--lane_keep_strength",type=float, default=0.15,
                   help="Max correction added to steering (0=off, 0.15=gentle)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"No checkpoint: {args.checkpoint}")

    ckpt  = torch.load(args.checkpoint, map_location=device)
    model = DrivingCNN().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"\n  Loaded epoch {ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss',0):.5f}")
    print(f"  smooth={args.smooth}  centre_pull={args.centre_pull}  "
          f"curve_exit_thresh={args.curve_exit_thresh}")
    print(f"  max_speed={args.max_speed}  max_thr={args.max_thr}  "
          f"steer_scale={args.steer_scale}")
    print(f"  lane_keep_zone={args.lane_keep_zone}m  "
          f"lane_keep_strength={args.lane_keep_strength}")

    env = make_env(args.seed)

    print(f"\n{'='*60}")
    print(f"  Autonomous Driving — {args.episodes} episodes")
    print(f"  Lane monitoring: ON  (spawns middle lane, index 1)")
    print(f"  Lane keeping  : ON  (corrects drift toward edges)")
    print(f"{'='*60}\n")

    all_steps        = []
    all_offroad      = []
    all_crash        = []
    all_lane_changes = []

    for ep in range(1, args.episodes + 1):
        obs, info  = env.reset()
        prev_steer = 0.0
        smooth_st  = 0.0
        total_r    = 0.0
        offroad    = False
        crash      = False
        t0         = time.time()

        prev_lane         = get_lane_index(env)
        lane_changes      = 0
        lane_change_steps = []

        print(f"  ── Episode {ep}/{args.episodes}  "
              f"(spawn lane={prev_lane}) ──────────────────")

        for step in range(3000):
            if "image" not in obs:
                obs, _, term, trunc, info = env.step([0, 0])
                if term or trunc: break
                continue

            speed = float(getattr(env.agent, "speed", 0.0))

            # ── model inference ───────────────────────────────────────────────
            raw_steer, throttle, brake = get_raw(
                model, obs, speed, prev_steer, device)
            raw_steer *= args.steer_scale
            smooth_st  = process_steering(
                raw_steer, smooth_st,
                args.smooth, args.centre_pull, args.curve_exit_thresh)

            # ── lane keeping correction ───────────────────────────────────────
            correction = lane_keep_correction(
                info, args.lane_keep_zone, args.lane_keep_strength)
            final_steer = smooth_st + correction

            throttle = min(throttle, args.max_thr)
            if speed > args.max_speed:
                throttle = 0.0

            steer = float(np.clip(final_steer, -1.0, 1.0))
            accel = float(np.clip(throttle - brake, -1.0, 1.0))

            obs, reward, term, trunc, info = env.step([steer, accel])
            total_r    += reward
            prev_steer  = smooth_st   # feed back pre-correction steering

            # ── lane change detection ─────────────────────────────────────────
            curr_lane = get_lane_index(env)
            if curr_lane != -1 and prev_lane != -1 and curr_lane != prev_lane:
                lane_changes += 1
                lane_change_steps.append(step)
                direction = "LEFT" if curr_lane < prev_lane else "RIGHT"
                corr_str  = f"  correction={correction:+.3f}" if abs(correction) > 0.01 else ""
                print(f"    ⚠ LANE CHANGE step={step:4d}  "
                      f"lane {prev_lane}→{curr_lane} ({direction})  "
                      f"speed={speed:.1f}  steer={steer:+.3f}{corr_str}")
            prev_lane = curr_lane if curr_lane != -1 else prev_lane

            if info.get("crash",       False): crash   = True
            if info.get("out_of_road", False): offroad = True

            if step % 100 == 0:
                lat_l = info.get("lateral_to_left",  99)
                lat_r = info.get("lateral_to_right", 99)
                lane_str  = f"lane={curr_lane}" if curr_lane != -1 else "lane=?"
                corr_str  = f"  corr={correction:+.3f}" if abs(correction) > 0.01 else ""
                edge_str  = f"  edge({lat_l:.1f}L/{lat_r:.1f}R)" \
                            if min(lat_l, lat_r) < args.lane_keep_zone else ""
                print(f"    step={step:4d}  speed={speed:4.1f}  "
                      f"raw={raw_steer:+.3f}  smooth={smooth_st:+.3f}  "
                      f"thr={throttle:.3f}  {lane_str}{corr_str}{edge_str}")

            if term or trunc:
                break

        elapsed = time.time() - t0
        all_steps.append(step + 1)
        all_offroad.append(int(offroad))
        all_crash.append(int(crash))
        all_lane_changes.append(lane_changes)

        result = "ARRIVED ✔" if not offroad and not crash else \
                 ("off-road ✗" if offroad else "crash ✗")
        print(f"\n  Episode {ep} summary:")
        print(f"    Result       : {result}")
        print(f"    Steps        : {step+1}")
        print(f"    Reward       : {total_r:.1f}")
        print(f"    Lane changes : {lane_changes}"
              + (f"  at steps {lane_change_steps}" if lane_change_steps else ""))
        print(f"    Time         : {elapsed:.0f}s\n")

    env.close()

    print(f"{'='*60}")
    print(f"  FINAL RESULTS — {args.episodes} episodes")
    print(f"  Avg steps      : {np.mean(all_steps):.0f}")
    print(f"  Offroad rate   : {np.mean(all_offroad)*100:.0f}%")
    print(f"  Crash rate     : {np.mean(all_crash)*100:.0f}%")
    print(f"  Avg lane chgs  : {np.mean(all_lane_changes):.1f} per episode")
    print(f"  Total lane chgs: {sum(all_lane_changes)}")
    print(f"{'='*60}")
    if sum(all_lane_changes) > 0:
        print()
        print("  Lane changes still occurring? Try:")
        print("    --lane_keep_strength 0.25   (stronger correction)")
        print("    --lane_keep_zone 1.5        (trigger correction earlier)")


if __name__ == "__main__":
    main()
