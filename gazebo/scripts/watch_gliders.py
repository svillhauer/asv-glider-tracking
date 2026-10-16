"""
Live glider check. Run inside the container while the stable bridge is running:
    python3 /work/scripts/watch_gliders.py            # 20 km run (300 s ticks)
    python3 /work/scripts/watch_gliders.py --tick-s 30   # 2 km run

Prints one line per env tick from /asv_stable/metrics: glider depth, each glider's
horizontal speed since the previous tick, their separation, and the ASV's slant range
to each. Expected with the default DAVE gliders and no current: speeds ~0.30 m/s
(lower on the tick a glider reaches the surface, 0 while it sits there), depth changing
~42 m per 300 s tick between 0 and 1000 m, separation 20,000 m.
"""
import argparse
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class Watch(Node):
    def __init__(self, tick_s):
        super().__init__("watch_gliders")
        self.tick_s = tick_s
        self.prev = None
        self.create_subscription(Float32MultiArray, "/asv_stable/metrics", self.cb, 10)
        print(f"{'tick':>5} {'depth m':>8} {'g1 m/s':>7} {'g2 m/s':>7} {'sep m':>8} "
              f"{'slant1 m':>9} {'slant2 m':>9} {'in range':>8}", flush=True)

    def cb(self, msg):
        m = dict(zip(msg.layout.dim[0].label.split(","), msg.data, strict=False))
        g1, g2 = (m["g1_x_m"], m["g1_y_m"]), (m["g2_x_m"], m["g2_y_m"])
        v1 = v2 = float("nan")
        if self.prev is not None and m["step"] > self.prev[0]:
            dt = (m["step"] - self.prev[0]) * self.tick_s
            v1, v2 = math.dist(g1, self.prev[1]) / dt, math.dist(g2, self.prev[2]) / dt
        self.prev = (m["step"], g1, g2)
        print(f"{int(m['step']):5d} {m['depth_m']:8.0f} {v1:7.3f} {v2:7.3f} {math.dist(g1, g2):8.0f} "
              f"{m['slant_range1_m']:9.0f} {m['slant_range2_m']:9.0f} {m['in_range_frac']:8.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tick-s", type=float, default=300.0, help="env tick length [s] (300 at 20 km, 30 at 2 km)")
    args = ap.parse_args()
    rclpy.init()
    try:
        rclpy.spin(Watch(args.tick_s))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
