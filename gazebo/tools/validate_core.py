"""
Validate the stable Gazebo bridge against the training env BEFORE using Gazebo.

Run on the host from the repository root (needs stable-baselines3):
    python3 gazebo/tools/validate_core.py

Checks, all closed-loop with the trained policy acting:
  1. step_gazebo(): feeding it the env's own ASV pose reproduces the env's
     observation and reward exactly (the Gazebo path is the training code).
  2. The Gazebo frame round trip (env frame -> metres relative to the WAM-V
     spawn -> env frame) changes the observation only by float rounding.
  3. The numpy actor + numpy VecNormalize match SB3's deterministic predict.
  4. Episode return through the bridge equals the env's episode return.
  5. Kinematic mode (step_kinematic, used without Gazebo) with the env glider
     model reproduces env.step exactly.
  6. DAVE glider model (glider_model="dave") at 20 km separation: the ported flight
     model's glide speeds, the dive profile and period, constant separation with no
     current, the DAC recovered from a uniform current, and closed-loop policy returns
     with DAVE gliders vs the training glider model under the same (zero) current.
"""
import sys
from pathlib import Path

import numpy as np

GAZEBO = Path(__file__).resolve().parents[1]
POLICY = GAZEBO / "policy"
MODEL = GAZEBO.parent / "models" / "two_glider" / "policy.zip"   # same weights as the shipped policy
sys.path.insert(0, str(GAZEBO / "ros_ws" / "src" / "asv_bridge_stable"))

from asv_bridge_stable.ekf_core_stable import STABLE_CONFIG, EkfCoreStable, physical_scales  # noqa: E402
from asv_bridge_stable.policy_numpy import SACActorNumpy, VecNormalizeNumpy  # noqa: E402
from asv_bridge_stable.stable_env import AsvNGliderCurrentsADCPEnvKnownDepthFlatStable as Env  # noqa: E402

SEEDS = [900, 907, 77]
SEP_M = 2000.0
N_STEPS = 2880          # one full episode per seed

actor = SACActorNumpy(str(POLICY / "actor_weights.npz"))
vecnorm = VecNormalizeNumpy(str(POLICY / "best_vec_normalize_arrays.npz"))

ok = True


def report(name, value, tol):
    global ok
    passed = value <= tol
    ok &= passed
    print(f"  {name:58s} {value:.2e}  (tol {tol:.0e})  {'OK' if passed else 'FAIL'}")


print(f"Stable reference -> Gazebo bridge validation  (sep_m={SEP_M:.0f})")
for k, v in physical_scales(SEP_M).items():
    print(f"  {k:20s} {v:10.4f}")

# 3. numpy policy vs SB3
import pickle  # noqa: E402

from stable_baselines3 import SAC  # noqa: E402

sb3 = SAC.load(str(MODEL), device="cpu")
with open(POLICY / "best_vec_normalize.pkl", "rb") as f:
    vn = pickle.load(f)

print("\nPolicy export")
env = Env(dict(STABLE_CONFIG))
obs, _ = env.reset(seed=SEEDS[0])
d_norm, d_act = 0.0, 0.0
for _ in range(500):
    on_np = vecnorm(obs)
    on_sb3 = vn.normalize_obs(obs[None])[0]
    a_np = actor.predict(on_np)
    a_sb3, _ = sb3.predict(on_sb3, deterministic=True)
    d_norm = max(d_norm, float(np.abs(on_np - on_sb3).max()))
    d_act = max(d_act, float(np.abs(a_np - a_sb3).max()))
    obs, *_ = env.step(a_sb3)
report("numpy VecNormalize vs SB3 normalize_obs (max abs diff)", d_norm, 1e-5)
report("numpy actor vs SB3 predict(deterministic) (max abs diff)", d_act, 1e-4)

# 1, 2, 4. bridge core vs env
for seed in SEEDS:
    print(f"\nSeed {seed}")
    env = Env(dict(STABLE_CONFIG))
    obs_env, _ = env.reset(seed=seed)
    core_exact = EkfCoreStable(sep_m=SEP_M)          # fed the env's ASV pose directly
    core_frame = EkfCoreStable(sep_m=SEP_M)          # fed the pose via metres in the Gazebo frame
    obs_exact = core_exact.reset_core(seed=seed)
    obs_frame = core_frame.reset_core(seed=seed)

    d_exact = float(np.abs(obs_env - obs_exact).max())
    d_frame = float(np.abs(obs_env - obs_frame).max())
    d_rew = 0.0
    ret_env = ret_core = 0.0
    for _ in range(N_STEPS):
        a = actor.predict(vecnorm(obs_env))
        obs_env, r_env, _, trunc, _ = env.step(a)
        pos_m = core_frame.nd_to_gazebo(env.asv_pos)        # what GPS would report
        obs_exact = core_exact.step_gazebo(env.asv_pos, env.asv_heading, a)
        obs_frame = core_frame.step_gazebo(core_frame.gazebo_to_nd(pos_m), env.asv_heading, a)
        d_exact = max(d_exact, float(np.abs(obs_env - obs_exact).max()))
        d_frame = max(d_frame, float(np.abs(obs_env - obs_frame).max()))
        d_rew = max(d_rew, abs(r_env - core_exact.last_info["reward"]))
        ret_env += r_env
        ret_core += core_exact.last_info["reward"]
        if trunc:
            break
    report("step_gazebo(env pose) obs vs env obs (max abs diff)", d_exact, 1e-6)
    report("step_gazebo reward vs env reward (max abs diff)", d_rew, 1e-6)
    report("via Gazebo frame [m] obs vs env obs (max abs diff)", d_frame, 1e-4)
    print(f"  episode return  env={ret_env:,.0f}  bridge={ret_core:,.0f}")

# 5. kinematic mode with env gliders == env
print("\nKinematic mode (env gliders)")
env = Env(dict(STABLE_CONFIG))
obs_env, _ = env.reset(seed=SEEDS[0])
core = EkfCoreStable(sep_m=SEP_M, glider_model="env")
core.reset_core(seed=SEEDS[0])
d = 0.0
for _ in range(N_STEPS):
    a = actor.predict(vecnorm(obs_env))
    obs_env, *_ = env.step(a)
    d = max(d, float(np.abs(obs_env - core.step_kinematic(a)).max()))
report("step_kinematic obs vs env obs (max abs diff)", d, 1e-5)

# 6. DAVE gliders
import math  # noqa: E402

from asv_bridge_stable.dave_glider import DaveGlider, flight_velocity  # noqa: E402

SEP_DAVE = 20000.0
print(f"\nDAVE glider model  (sep_m={SEP_DAVE:.0f})")
core = EkfCoreStable(config={"current_max": 0.0}, sep_m=SEP_DAVE, glider_model="dave")
core.reset_core(seed=SEEDS[0])
cs = core.dave[0].cycle_stats()
pump = core.dave_cfg["dave_pump_cc"]
print(f"  pump {pump:.0f} cc, pitch {core.dave_cfg['dave_pitch_deg']:.0f} deg -> horizontal {cs['u_descend']:.3f} m/s, "
      f"vertical {-cs['w_descend']:.3f} m/s, glide {cs['glide_angle_deg']:.1f} deg, "
      f"cycle {cs['period_s']/3600:.2f} h to {core.dave_cfg['dave_max_depth_m']:.0f} m, "
      f"mean horizontal {cs['mean_horizontal_speed']:.3f} m/s")
report("glide speed vs training glider speed 0.30 m/s", abs(cs["u_descend"] - 0.30), 1e-9)
u_a, w_a = flight_velocity(+pump, -math.radians(26.0))
report("ascent mirrors descent (DAVE model symmetry)", abs(u_a - cs["u_descend"]) + abs(w_a + cs["w_descend"]), 1e-12)

# dive profile + separation, open loop (ASV parked), no current
depths, seps, surf_ticks = [], [], []
for k in range(600):
    core.step_gazebo(core.asv_pos, core.asv_heading, np.zeros(2, np.float32))
    depths.append(core._depth * SEP_DAVE)
    seps.append(np.linalg.norm(core.g_pos[0] - core.g_pos[1]) * SEP_DAVE)
    if core.glider_modes()[0] == "surface" and (not surf_ticks or surf_ticks[-1] != k - 1):
        surf_ticks.append(k)
period_ticks = np.diff(surf_ticks).mean() if len(surf_ticks) > 1 else float("nan")
print(f"  max depth {max(depths):.1f} m, surfacings at ticks {surf_ticks[:4]}..., "
      f"period {period_ticks * core.tick_s / 3600:.2f} h")
# the bridge samples depth once per 300 s tick, so check the turn-around depth on the glider itself
g = DaveGlider([0.0, 0.0], 0.0, 1000.0, math.radians(26.0), pump, 600.0)
turn = None
while g.mode != "surface":
    before = g.mode
    g.advance(5.0, np.zeros(2))
    if before == "descend" and g.mode == "ascend":
        turn = g.depth
report("dive turn-around depth vs 1000 m (5 s substeps)", abs(turn - 1000.0), 1e-9)
report("dive period vs nominal (h)", abs(period_ticks * core.tick_s - cs["period_s"]) / 3600, core.tick_s / 3600)
# positions are stored as float32 in nondimensional units: ~1e-7 x 20 km = 2 mm resolution
report("glider separation drift with no current (m)", max(seps) - min(seps), 1e-2)

# DAC from a uniform current
g = DaveGlider([0.0, 0.0], 0.3, 1000.0, math.radians(26.0), pump, 600.0)
cur = np.array([0.10, -0.05])
while g.dac is None:
    g.advance(5.0, cur)
report("DAC recovered from uniform 0.1/-0.05 m/s current (m/s)", float(np.abs(g.dac - cur).max()), 1e-9)

# continuous glider updates between ticks (what the bridge does for Gazebo) == one batch per tick
cb = EkfCoreStable(config={"current_max": 0.08}, sep_m=SEP_DAVE, glider_model="dave")
cc = EkfCoreStable(config={"current_max": 0.08}, sep_m=SEP_DAVE, glider_model="dave")
ob, oc = cb.reset_core(seed=SEEDS[1]), cc.reset_core(seed=SEEDS[1])
rng = np.random.default_rng(0)
d = 0.0
for _ in range(600):
    a = actor.predict(vecnorm(ob))
    for t in np.sort(rng.uniform(0, cc.tick_s * 1.1, size=rng.integers(0, 40))):   # incl. late timer calls
        cc.advance_gliders_to(t)
    ob, oc = cb.step_kinematic(a), cc.step_kinematic(a)
    d = max(d, float(np.abs(ob - oc).max()), float(np.abs(cb.g_pos - cc.g_pos).max()))
report("continuous glider updates vs per-tick batch (max abs diff, with current)", d, 0.0)

# closed loop: training gliders vs DAVE gliders, same seeds, zero current
for seed in SEEDS:
    rets = {}
    for model in ("env", "dave"):
        c = EkfCoreStable(config={"current_max": 0.0}, sep_m=SEP_DAVE, glider_model=model)
        o = c.reset_core(seed=seed)
        ret, inr = 0.0, 0.0
        for _ in range(N_STEPS):
            o = c.step_kinematic(actor.predict(vecnorm(o)))
            ret += c.last_info["reward"]
            inr += c.last_info["in_range_frac"]
        rets[model] = (ret, inr / N_STEPS)
    print(f"  seed {seed}: return  env gliders {rets['env'][0]:,.0f} (in range {rets['env'][1]:.2f})   "
          f"DAVE gliders {rets['dave'][0]:,.0f} (in range {rets['dave'][1]:.2f})")

# 7. sensing realism: realistic acoustic channel and noisy ADCP
print(f"\nSensing realism  (sep_m={SEP_DAVE:.0f}, DAVE gliders, training current)")
REAL = {"acoustic_model": "realistic", "adcp_noise": True}
c = EkfCoreStable(config={"current_max": 0.08, **REAL}, sep_m=SEP_DAVE, glider_model="dave")
o = c.reset_core(seed=SEEDS[0])
dist, rx, adcp_err = [], [], []
for _ in range(N_STEPS):
    o = c.step_kinematic(actor.predict(vecnorm(o)))
    dist.append(c.last_info["dist_each"])
    rx.append(c.last_info["received"])
    adcp_err.append((c.adcp_reading - np.asarray(c._get_current(c.asv_pos))) * 2.5)
dist, rx, adcp_err = np.concatenate(dist) / c.comm_range, np.concatenate(rx), np.array(adcp_err)
worst = 0.0
for lo, hi in ((0.0, 0.8), (0.8, 0.95), (0.95, 1.05), (1.05, 1.2)):
    m = (dist >= lo) & (dist < hi)
    if m.sum() < 30:
        continue
    pred = float(np.mean(c.reception_probability(dist[m] * c.comm_range)))
    emp = float(rx[m].mean())
    se = np.sqrt(pred * (1 - pred) / m.sum())
    worst = max(worst, abs(emp - pred) / max(se, 1e-9))
    print(f"  slant/R_comm {lo:.2f}-{hi:.2f}: received {emp:.3f}  model {pred:.3f}  (n={m.sum()})")
report("reception rate vs model, worst deviation (standard errors)", worst, 3.0)
print(f"  sound-speed scale this episode {c.sound_speed_scale:.4f}; ADCP bias {(c.adcp_bias * 2.5).round(4)} m/s")
report("ADCP error mean vs drawn bias (m/s)", float(np.abs(adcp_err.mean(0) - c.adcp_bias * 2.5).max()), 0.005)
report("ADCP error std vs 0.03 m/s", float(np.abs(adcp_err.std(0) - 0.03).max()), 0.005)

# realism must not change the mission: same glider motion and current field as the ideal run
ci = EkfCoreStable(config={"current_max": 0.08}, sep_m=SEP_DAVE, glider_model="dave")
cr = EkfCoreStable(config={"current_max": 0.08, **REAL}, sep_m=SEP_DAVE, glider_model="dave")
ci.reset_core(seed=SEEDS[1])
cr.reset_core(seed=SEEDS[1])
for _ in range(300):
    a = np.array([0.5, 0.1], np.float32)
    ci.step_kinematic(a)
    cr.step_kinematic(a)
report("glider tracks, ideal vs realistic sensing, same actions (m)",
       float(np.abs(ci.g_pos - cr.g_pos).max() * SEP_DAVE), 1e-6)

# closed-loop effect of each realism option (Gym, 20 km, DAVE gliders, training current)
print("  closed-loop, mean over seeds:  reward/step | in range % | median EKF unc. (m) | median true error (m)")
for name, extra in (("ideal", {}), ("acoustic", {"acoustic_model": "realistic"}),
                    ("ADCP", {"adcp_noise": True}), ("acoustic + ADCP", REAL)):
    res = []
    for seed in SEEDS:
        c = EkfCoreStable(config={"current_max": 0.08, **extra}, sep_m=SEP_DAVE, glider_model="dave")
        o = c.reset_core(seed=seed)
        r = []
        inr = []
        unc = []
        err = []
        for _ in range(N_STEPS):
            o = c.step_kinematic(actor.predict(vecnorm(o)))
            r.append(c.last_info["reward"])
            inr.append(c.last_info["in_range_frac"])
            unc.append(np.asarray(c.last_info["det_P_each"], float) ** 0.25 * SEP_DAVE)
            err.append(np.linalg.norm(c.g_pos - c.g_est_pos, axis=1) * SEP_DAVE)
        res.append((np.mean(r), 100 * np.mean(inr), np.median(unc), np.median(err)))
    m = np.mean(res, axis=0)
    print(f"    {name:16s} {m[0]:7.2f}      {m[1]:6.1f}        {m[2]:8.0f}            {m[3]:8.0f}")

# 8. current vs depth and glider navigation errors
from asv_bridge_stable.dave_glider import DaveGlider  # noqa: E402

print(f"\nCurrent vs depth and glider navigation  (sep_m={SEP_DAVE:.0f}, DAVE gliders)")
DECAY = {"current_profile": "decay"}
c = EkfCoreStable(config={"current_max": 0.08, **DECAY}, sep_m=SEP_DAVE, glider_model="dave")
s0, a0 = c.current_profile_factor(0.0)
report("profile at the surface: |scale - 1| + |rotation|", abs(s0 - 1) + abs(a0), 1e-12)
for z in (100, 250, 500, 1000):
    s, a = c.current_profile_factor(z)
    print(f"  depth {z:5d} m: x{s:.3f}, rotated {np.degrees(a):+.1f} deg")


def one_dive(core, glider, surface_c):
    """Integrate one full yo-yo in a horizontally uniform current; returns (DAC, mean current over the dive)."""
    seen = []
    for _ in range(int(6 * 3600 / 5)):
        cur = surface_c if core is None else core_profile(core, surface_c, glider.depth)
        if glider.mode != glider.SURFACE:
            seen.append(cur)
        glider.advance(5.0, cur)
        if glider.surfaced_event:
            return glider.dac, np.mean(seen, axis=0)
    raise RuntimeError("glider never surfaced")


def core_profile(core, surface_c, depth):
    s, a = core.current_profile_factor(depth)
    return s * np.array([np.cos(a) * surface_c[0] - np.sin(a) * surface_c[1],
                         np.sin(a) * surface_c[0] + np.cos(a) * surface_c[1]])


cs = DaveGlider([0, 0], 0.7, 1000.0, np.radians(26), c.dave_cfg["dave_pump_cc"], 600.0).cycle_stats()
surface_c = np.array([0.15, -0.10])
g = DaveGlider([0, 0], 0.7, 1000.0, np.radians(26), c.dave_cfg["dave_pump_cc"], 600.0)
g.realistic_nav, g.nav_rng = True, np.random.default_rng(0)
g.dive_start_fix()
dac, mean_c = one_dive(c, g, surface_c)
report("decay profile, error-free nav: DAC vs mean current over the dive (m/s)",
       float(np.abs(dac - mean_c).max()), 1e-9)
ratio = np.linalg.norm(dac) / np.linalg.norm(surface_c)
print(f"  surface current {surface_c} m/s -> DAC {dac.round(4)} m/s (x{ratio:.2f})")

# navigation errors: DAC = mean current + (I - s R(b)) * mean glide velocity, plus GPS noise / dive time
b, sc = np.radians(2.0), 1.05
g = DaveGlider([0, 0], 0.7, 1000.0, np.radians(26), c.dave_cfg["dave_pump_cc"], 600.0)
g.realistic_nav, g.nav_heading_bias, g.nav_speed_scale, g.gps_std_m = True, b, sc, 5.0
g.nav_rng = np.random.default_rng(1)
g.dive_start_fix()
dac, mean_c = one_dive(None, g, surface_c)
v = cs["mean_horizontal_speed"] * cs["period_s"] / (cs["period_s"] - 600.0) * np.array([np.cos(0.7), np.sin(0.7)])
R = np.array([[np.cos(b), -np.sin(b)], [np.sin(b), np.cos(b)]])
pred = mean_c + (np.eye(2) - sc * R) @ v
gps_tol = 4 * 5.0 * np.sqrt(2) / (cs["period_s"] - 600.0)
report("nav errors: DAC vs predicted bias, minus GPS-noise allowance (m/s)",
       max(0.0, float(np.abs(dac - pred).max()) - gps_tol), 1e-9)
print(f"  compass +2.0 deg, speed +5%: DAC error {(dac - mean_c).round(4)} m/s "
      f"(|err| {np.linalg.norm(dac - mean_c):.4f} m/s vs glide {np.linalg.norm(v):.2f} m/s)")

# neither option may change the ASV-side mission (current field at the surface, glider layout)
ci = EkfCoreStable(config={"current_max": 0.08}, sep_m=SEP_DAVE, glider_model="dave")
cn = EkfCoreStable(config={"current_max": 0.08, "glider_nav_error": True}, sep_m=SEP_DAVE, glider_model="dave")
cd = EkfCoreStable(config={"current_max": 0.08, **DECAY}, sep_m=SEP_DAVE, glider_model="dave")
oi, on, od = ci.reset_core(seed=SEEDS[1]), cn.reset_core(seed=SEEDS[1]), cd.reset_core(seed=SEEDS[1])
report("reset observation, nav errors / decay vs training current",
       float(max(np.abs(oi - on).max(), np.abs(oi - od).max())), 0.0)
for _ in range(300):
    a = np.array([0.5, 0.1], np.float32)
    ci.step_kinematic(a)
    cn.step_kinematic(a)
    cd.step_kinematic(a)
report("glider tracks, nav errors on vs off, same actions (m)",
       float(np.abs(ci.g_pos - cn.g_pos).max() * SEP_DAVE), 1e-6)
report("ASV track, decay vs uniform, same actions (m)", float(np.abs(ci.asv_pos - cd.asv_pos).max() * SEP_DAVE), 1e-6)
print(f"  glider displacement after 300 ticks, decay vs uniform: "
      f"{(np.linalg.norm(cd.g_pos - ci.g_pos, axis=1) * SEP_DAVE).round(0)} m")

print("  closed-loop, mean over seeds:  reward/step | in range % | median EKF unc. (m) | median true error (m)")
FULL = {**REAL, **DECAY, "glider_nav_error": True}
for name, extra in (("training current", {}), ("decay", DECAY), ("nav errors", {"glider_nav_error": True}),
                    ("decay + nav", {**DECAY, "glider_nav_error": True}), ("all realism", FULL)):
    res = []
    for seed in SEEDS:
        c = EkfCoreStable(config={"current_max": 0.08, **extra}, sep_m=SEP_DAVE, glider_model="dave")
        o = c.reset_core(seed=seed)
        r = []
        inr = []
        unc = []
        err = []
        for _ in range(N_STEPS):
            o = c.step_kinematic(actor.predict(vecnorm(o)))
            r.append(c.last_info["reward"])
            inr.append(c.last_info["in_range_frac"])
            unc.append(np.asarray(c.last_info["det_P_each"], float) ** 0.25 * SEP_DAVE)
            err.append(np.linalg.norm(c.g_pos - c.g_est_pos, axis=1) * SEP_DAVE)
        res.append((np.mean(r), 100 * np.mean(inr), np.median(unc), np.median(err)))
    m = np.mean(res, axis=0)
    print(f"    {name:16s} {m[0]:7.2f}      {m[1]:6.1f}        {m[2]:8.0f}            {m[3]:8.0f}")

print("\nPASS" if ok else "\nFAIL")
sys.exit(0 if ok else 1)
