"""
Underwater glider motion from Project DAVE's kinematic flight model.

Source: Field-Robotics-Lab/glider_hybrid_whoi, kinematics_ros_plugins/src/KinematicsROSPlugin.cc,
KinematicsROSPlugin::ConveyKinematicsCommands (commit a855760, Apache-2.0,
Copyright 2020 Naval Postgraduate School). The flight-model block is ported line
for line in ``flight_velocity``; the thruster term is dropped (a Slocum glider,
motor power 0, so v_thrust = 0 in DAVE as well).

DAVE's model, for pumped volume V [cc] and commanded pitch theta [rad]
(ENU / REP-103, positive pitch = nose down):

    m0     = V * 1e-6 * rho                              net buoyancy mass [kg]
    alpha  = -3 deg if theta > 0 else +3 deg             angle of attack
    xi     = theta - alpha
    V > 0 (ascend):  vk = sqrt( 2 m0 g / (rho A)),  vz =  vk C_D / C_L / sqrt(C_L),
                     vx = vk sqrt(-sin(xi) / C_D) cos(xi)   if -sin(xi) > 0 else 0
    V < 0 (descend): vk = sqrt(-2 m0 g / (rho A)),  vz = -vk C_D / C_L / sqrt(C_L),
                     vx = vk sqrt( sin(xi) / C_D) cos(xi)   if  sin(xi) > 0 else 0
    horizontal speed  u = vx cos(theta) + vz sin(theta)  (along the heading)
    vertical speed    w = vx sin(theta) + vz cos(theta)  (positive up)

DAVE applies this as the vehicle's through-water velocity and adds the ocean
current on top (applyOceanCurrent). At the surface its CheckSubmergence holds a
glider that is still commanded to rise at the surface.

What this module adds around the flight model (DAVE leaves these to external
mission scripts): a yo-yo dive controller (descend to max_depth, climb to the
surface, stay at the surface for surface_time_s, repeat), constant heading,
and the glider's own dead reckoning with a GPS fix at each surfacing, from which
the depth-averaged current (DAC) is computed.

Parameters: C_D, C_L, rho, g are DAVE's values. A = 0.0345 m^2 is the default in
DAVE's plugin source (a Slocum hull's frontal area, 21 cm diameter); DAVE's example
xacro passes 0.345, which looks like a factor-10 slip and gives unrealistically slow
gliders (~0.08 m/s), so the source default is used.
"""

import math

import numpy as np

RHO = 1024.0     # fluid density [kg/m^3]      (DAVE default)
G = 9.81         # gravity [m/s^2]              (DAVE default)
AREA = 0.0345    # reference area [m^2]         (DAVE plugin source default)
C_D = 0.2534     # drag coefficient             (DAVE)
C_L = 0.4160     # lift coefficient             (DAVE)


def flight_velocity(pumped_volume_cc: float, pitch: float) -> tuple[float, float]:
    """DAVE's flight model: (horizontal speed along heading, vertical speed up) [m/s]."""
    if pumped_volume_cc == 0.0:
        return 0.0, 0.0
    m_0 = pumped_volume_cc / 1_000_000.0 * RHO
    alpha = -3.0 / 180.0 * math.pi if pitch > 0.0 else 3.0 / 180.0 * math.pi
    xi = pitch - alpha
    if m_0 > 0:
        v_kernel = math.sqrt((2 * m_0 * G) / (RHO * AREA))
        vz = v_kernel * C_D / C_L / math.sqrt(C_L)
        vx = v_kernel * math.sqrt(-math.sin(xi) / C_D) * math.cos(xi) if -math.sin(xi) / C_D > 0 else 0.0
    else:
        v_kernel = math.sqrt((-2 * m_0 * G) / (RHO * AREA))
        vz = -v_kernel * C_D / C_L / math.sqrt(C_L)
        vx = v_kernel * math.sqrt(math.sin(xi) / C_D) * math.cos(xi) if math.sin(xi) / C_D > 0 else 0.0
    u = vx * math.cos(pitch) + vz * math.sin(pitch)
    w = vx * math.sin(pitch) + vz * math.cos(pitch)
    return u, w


def pump_volume_for_speed(target_u: float, pitch: float) -> float:
    """Pumped volume [cc] whose DAVE glide gives horizontal speed target_u at the given |pitch|.

    Speeds scale with sqrt(|V|), so one evaluation fixes the constant.
    """
    u_ref, _ = flight_velocity(-100.0, abs(pitch))
    return 100.0 * (target_u / u_ref) ** 2


class DaveGlider:
    """One glider: DAVE flight model + yo-yo controller + dead reckoning. SI units, ENU."""

    DESCEND, ASCEND, SURFACE = "descend", "ascend", "surface"

    def __init__(self, pos_m, heading, max_depth_m, pitch_rad, pump_cc, surface_time_s):
        self.pos = np.asarray(pos_m, dtype=np.float64).copy()   # horizontal position [m]
        self.depth = 0.0                                        # [m], positive down
        self.heading = float(heading)                           # ENU yaw [rad]
        self.max_depth = float(max_depth_m)
        self.pitch_mag = abs(float(pitch_rad))
        self.pump = abs(float(pump_cc))
        self.surface_time = float(surface_time_s)
        self.mode = self.DESCEND
        self.surface_timer = 0.0
        self.pitch = self.pitch_mag
        # Glider's own navigation: dead reckoning from its flight model, GPS fix at the surface.
        self.dr_pos = self.pos.copy()
        self.t_since_fix = 0.0
        self.dac = None            # last depth-averaged current [m/s], None until the first dive ends
        self.surfaced_event = False
        # Navigation errors (set by the core when glider_nav_error is on). With realistic_nav the
        # glider dead-reckons with a compass bias and a speed error, takes a noisy GPS fix at
        # surfacing and another before diving, and computes the DAC over the dive only.
        self.realistic_nav = False
        self.nav_heading_bias = 0.0   # [rad]
        self.nav_speed_scale = 1.0
        self.gps_std_m = 0.0
        self.nav_rng = None

    def through_water_velocity(self):
        """(vx_east, vy_north, w_up) [m/s] from DAVE's flight model for the current mode."""
        if self.mode == self.SURFACE:
            self.pitch = 0.0
            return 0.0, 0.0, 0.0
        if self.mode == self.DESCEND:
            self.pitch = +self.pitch_mag
            u, w = flight_velocity(-self.pump, self.pitch)
        else:
            self.pitch = -self.pitch_mag
            u, w = flight_velocity(+self.pump, self.pitch)
        return u * math.cos(self.heading), u * math.sin(self.heading), w

    def advance(self, dt_s, current_mps):
        """Integrate dt_s seconds. current_mps: horizontal ocean current at the glider [m/s]."""
        self.surfaced_event = False
        vx, vy, w = self.through_water_velocity()
        self.pos += (np.array([vx, vy]) + current_mps) * dt_s
        if self.realistic_nav:
            # the glider's belief: heading off by the compass bias, speed off by the flight-model error
            u = math.hypot(vx, vy) * self.nav_speed_scale
            h = self.heading + self.nav_heading_bias
            if self.mode != self.SURFACE:
                self.dr_pos += np.array([u * math.cos(h), u * math.sin(h)]) * dt_s
                self.t_since_fix += dt_s
        else:
            self.dr_pos += np.array([vx, vy]) * dt_s      # dead reckoning knows only its own glide
            self.t_since_fix += dt_s
        self.depth -= w * dt_s

        if self.mode == self.DESCEND and self.depth >= self.max_depth:
            self.depth = self.max_depth
            self.mode = self.ASCEND
        elif self.mode == self.ASCEND and self.depth <= 0.0:
            self.depth = 0.0
            self.mode = self.SURFACE
            self.surface_timer = 0.0
            self._gps_fix()
        elif self.mode == self.SURFACE:
            self.surface_timer += dt_s
            if self.surface_timer >= self.surface_time:
                self.mode = self.DESCEND
                if self.realistic_nav:
                    self.dive_start_fix()

    def _noisy_fix(self):
        return self.pos + self.nav_rng.normal(0.0, self.gps_std_m, size=2)

    def dive_start_fix(self):
        """Realistic navigation: GPS fix just before diving; dead reckoning restarts from it."""
        self.dr_pos = self._noisy_fix()
        self.t_since_fix = 0.0

    def _gps_fix(self):
        """Surfacing: GPS gives the true position; the DR error over the dive is the DAC."""
        if self.realistic_nav:
            # DAC over the dive only (surface drift excluded), from noisy fixes and biased DR
            if self.t_since_fix > 0:
                self.dac = (self._noisy_fix() - self.dr_pos) / self.t_since_fix
            self.t_since_fix = 0.0
            self.surfaced_event = True
            return
        if self.t_since_fix > 0:
            self.dac = (self.pos - self.dr_pos) / self.t_since_fix
        self.dr_pos = self.pos.copy()
        self.t_since_fix = 0.0
        self.surfaced_event = True

    def cycle_stats(self):
        """Nominal descent/ascent speeds and dive period for this configuration."""
        u_d, w_d = flight_velocity(-self.pump, +self.pitch_mag)
        u_a, w_a = flight_velocity(+self.pump, -self.pitch_mag)
        t_d, t_a = self.max_depth / -w_d, self.max_depth / w_a
        period = t_d + t_a + self.surface_time
        return {"u_descend": u_d, "w_descend": w_d, "u_ascend": u_a, "w_ascend": w_a,
                "glide_angle_deg": math.degrees(math.atan2(-w_d, u_d)),
                "period_s": period, "mean_horizontal_speed": (u_d * t_d + u_a * t_a) / period}
