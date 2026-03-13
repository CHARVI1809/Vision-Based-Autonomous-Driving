"""
debug_idm.py  —  Run this FIRST to find where IDM stores its action.
Prints every relevant attribute from the agent and its policy for 20 steps.

Usage:
  python debug_idm.py
"""
from metadrive import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.component.map.pg_map import MapGenerateMethod

MAP_CFG = dict(
    type   = MapGenerateMethod.PG_MAP_FILE,
    config = [
        None,
        {"id": "S", "pre_block_socket_index": 0, "length": 100},
        {"id": "C", "pre_block_socket_index": 0, "length": 60, "angle": 45},
        {"id": "S", "pre_block_socket_index": 0, "length": 80},
    ],
    lane_num=3, lane_width=3.5,
)

env = MetaDriveEnv(dict(
    use_render              = False,
    manual_control          = False,
    traffic_density         = 0.0,
    map_config              = MAP_CFG,
    num_scenarios           = 5,
    start_seed              = 100,
    image_observation       = False,   # off for speed
    agent_policy            = IDMPolicy,
    horizon                 = 500,
))

obs, info = env.reset()
print("\n=== IDM Action Source Finder ===\n")

for step in range(60):
    obs, reward, term, trunc, info = env.step([0, 0])
    ego    = env.agent
    policy = getattr(ego, "policy", None)
    speed  = float(getattr(ego, "speed", 0))

    if step % 10 == 0:
        print(f"--- step {step}  speed={speed:.2f} ---")

        # Check every likely attribute name
        for attr in ["before_step_action", "last_action", "action",
                     "current_action", "steering", "throttle_brake"]:
            val = getattr(ego, attr, "NOT FOUND")
            print(f"  ego.{attr:25s} = {val}")

        if policy:
            print(f"  policy type: {type(policy).__name__}")
            for attr in ["action", "last_action", "current_action",
                         "steering", "before_step_action", "target_speed"]:
                val = getattr(policy, attr, "NOT FOUND")
                print(f"  policy.{attr:22s} = {val}")

        # Check info dict for action keys
        action_keys = [k for k in info.keys()
                       if any(x in k.lower()
                              for x in ["steer","action","throttle","accel"])]
        if action_keys:
            print(f"  info action keys: {action_keys}")
            for k in action_keys:
                print(f"    info['{k}'] = {info[k]}")
        print()

    if term or trunc:
        obs, info = env.reset()

env.close()
print("Done. Check which attribute has non-zero values.")
