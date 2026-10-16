#!/usr/bin/env bash
# Headless batch of Gazebo missions in the full realistic setup: a fresh Gazebo per seed, every topic
# recorded, the recorder stopped cleanly at the end of the episode, and the bag extracted.
#
# Run from the HOST (no GUI needed; each seed gets its own container-internal Gazebo):
#   docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -v $(pwd)/gazebo:/work glidertrack-gazebo:jazzy \
#       bash /work/scripts/run_seeds.sh 900 901 902 903 904
#
# Settings (environment variables, all optional):
#   TICKS=150                               episode length in 300 s ticks (150 = 12.5 h)
#   TAG=realistic                           bag name: stable_dave_20km_<TAG>_s<seed>
#   WORLD=/work/worlds/rough/open_ocean.sdf  rough world (wind 8 m/s, Hs ~1.3 m); calm = /work/worlds/open_ocean.sdf
#   BRIDGE_ARGS="..."                       launch arguments for the bridge (default: full realism below)
# A seed whose bag folder already exists is skipped, so a stopped batch can be restarted.

source /opt/ros/jazzy/setup.bash
source /work/vrx_ws/install/setup.bash
source /work/ros_ws/install/setup.bash
export GZ_SIM_RESOURCE_PATH=/work/models:/work/worlds
set -u
# Job control: each background job gets its own process group (so it can be stopped as a group) and
# keeps SIGINT; without it, bash starts background jobs with SIGINT ignored and the recorder cannot
# be stopped cleanly (its bag is then never closed).
set -m

TICKS=${TICKS:-150}
TAG=${TAG:-realistic}
WORLD=${WORLD:-/work/worlds/rough/open_ocean.sdf}
BRIDGE_ARGS=${BRIDGE_ARGS:-"current_max:=0.08 acoustic_model:=realistic adcp_noise:=true current_profile:=decay glider_nav_error:=true"}
TOPICS="/asv_stable/metrics /asv_stable/action /asv_stable/obs_raw /asv_stable/water_current /asv_stable/adcp_current \
/wamv/sensors/gps/gps/fix /wamv/sensors/imu/imu/data"

stop_group() {   # SIGINT a process group, then SIGKILL whatever is left after $2 seconds
    kill -INT -"$1" 2>/dev/null
    for _ in $(seq 1 "$2"); do kill -0 -"$1" 2>/dev/null || return 0; sleep 1; done
    kill -KILL -"$1" 2>/dev/null
}

for SEED in "$@"; do
    NAME=stable_dave_20km_${TAG}_s${SEED}
    OUT=/work/bags/$NAME
    LOG=/work/bags/logs/$NAME
    if [ -e "$OUT" ]; then echo "[$NAME] bag exists, skipping"; continue; fi
    mkdir -p "$LOG"
    export ROS_DOMAIN_ID=$((40 + SEED % 50)) GZ_PARTITION=batch_$SEED
    echo "[$NAME] $(date '+%F %T')  world=$WORLD  ticks=$TICKS  args: $BRIDGE_ARGS"

    ros2 launch vrx_gz vrx_environment.launch.py world:="$WORLD" sim_mode:=full \
        config_file:=/work/config/wamv_spawn.yaml headless:=true > "$LOG/gazebo.log" 2>&1 &
    GZ=$!
    for _ in $(seq 1 150); do gz model -m wamv 2>/dev/null | grep -q "Model:" && break; sleep 2; done
    if ! gz model -m wamv 2>/dev/null | grep -q "Model:"; then
        echo "[$NAME] WAM-V never appeared, see $LOG/gazebo.log"; stop_group $GZ 20; continue
    fi
    sleep 15   # sensors and bridges up

    ros2 bag record -o "$OUT" --topics $TOPICS > "$LOG/record.log" 2>&1 &
    REC=$!
    sleep 5
    ros2 launch asv_bridge_stable stable_nondim.launch.py seed:=$SEED max_ticks:=$TICKS \
        $BRIDGE_ARGS > "$LOG/bridge.log" 2>&1 &
    BR=$!

    while ! grep -q "Episode finished" "$LOG/bridge.log"; do
        if ! kill -0 $BR 2>/dev/null || ! kill -0 $GZ 2>/dev/null; then
            echo "[$NAME] bridge or Gazebo exited early, see $LOG/"; break
        fi
        sleep 30
    done
    grep -E "step= *[0-9]+0 " "$LOG/bridge.log" | tail -1
    grep "Episode finished" "$LOG/bridge.log"

    stop_group $REC 60        # recorder first, so the bag is closed with every message
    stop_group $BR 20
    stop_group $GZ 30
    pkill -KILL -f "gz sim" 2>/dev/null; sleep 5

    python3 /work/scripts/extract_bag.py "$OUT" > "$LOG/extract.log" 2>&1 \
        && echo "[$NAME] $(date '+%F %T')  done: $(head -1 "$LOG/extract.log")" \
        || echo "[$NAME] extraction failed, see $LOG/extract.log"
done
