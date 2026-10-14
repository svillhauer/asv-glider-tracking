"""MP4 videos of missions: a top-down map over the current field, the glider dives, and the tracking error."""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
from matplotlib.patches import Circle, Ellipse  # noqa: E402
from PIL import Image  # noqa: E402

from .configs import LENGTH_M, TIME_S  # noqa: E402

GLIDER_COLORS = ["dodgerblue", "mediumorchid", "mediumseagreen", "darkorange", "crimson"]
GLIDER_LIGHT = ["#8FC7F5", "#DBAEEA", "#9ED9B8", "#F7C58F", "#F08C9C"]
ASV_COLOR = "navy"
TRACK_BG, PANEL_BG = "#f0f4f8", "#FBFCFE"
LOST_COLOR = "crimson"
STYLE = {
    "figure.facecolor": "white", "axes.facecolor": PANEL_BG, "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222", "axes.titleweight": "bold", "axes.labelweight": "bold", "axes.grid": True,
    "grid.color": "white", "grid.linewidth": 1.0, "xtick.color": "#333333", "ytick.color": "#333333",
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
}


@dataclass
class Mission:
    """Per-step arrays of one mission in nondimensional units (length = glider separation)."""

    asv: np.ndarray  # (T, 2)
    heading: np.ndarray  # (T,)
    gliders: np.ndarray  # (T, N, 2) true positions
    est: np.ndarray  # (T, N, 2) EKF positions
    cov: np.ndarray  # (T, N, 2, 2) EKF position covariance
    depth: np.ndarray  # (T,)
    in_range: np.ndarray  # (T, N) bool
    comm_range: float
    current: dict = field(default_factory=dict)  # velocities, stripe_width, num_stripes

    @classmethod
    def from_episode(cls, ep, env):
        return cls(asv=ep.asv, heading=ep.heading, gliders=ep.gliders, est=ep.estimates, cov=ep.covariances,
                   depth=ep.depth, in_range=ep.in_range, comm_range=ep.comm_range,
                   current=dict(velocities=env.current_velocities.copy(), stripe_width=env.current_stripe_width,
                                num_stripes=env.current_num_stripes))

    @property
    def true_error(self):
        return np.linalg.norm(self.gliders - self.est, axis=-1)

    @property
    def ekf_std(self):
        return np.sqrt(np.trace(self.cov, axis1=-2, axis2=-1))


def _icon(name):
    with resources.files("glidertrack").joinpath(f"assets/{name}.png").open("rb") as f:
        return Image.open(f).convert("RGBA").copy()


class _Map:
    """Top-down panel that follows the fleet."""

    def __init__(self, ax, m: Mission, km, speed_scale, half_window, trail_steps, icons=True):
        self.ax, self.m, self.km, self.half, self.trail = ax, m, km, half_window, trail_steps
        vel = np.asarray(m.current["velocities"]) * speed_scale
        self.vel, self.sw, self.ns = vel, m.current["stripe_width"], m.current["num_stripes"]
        self.cnorm = Normalize(0, max(float(np.hypot(*vel.T).max()), 1e-3))
        self.n = m.gliders.shape[1]
        self.icons = None
        if icons and self.n == 2:
            self.icons = {"asv": _icon("asv"), 0: _icon("glider_blue"), 1: _icon("glider_purple")}
        self.center = m.gliders.mean(axis=1)

    def _current_at(self, x):
        idx = np.clip(((x + self.ns * self.sw / 2.0) / self.sw).astype(int), 0, self.ns - 1)
        return self.vel[idx]

    def _place_icon(self, key, xy, angle_deg, zoom):
        im = self.icons[key].rotate(angle_deg, expand=True, resample=Image.BICUBIC)
        self.ax.add_artist(AnnotationBbox(OffsetImage(np.asarray(im), zoom=zoom), xy, frameon=False, zorder=8))

    def draw(self, k, title=None, legend=False):
        ax, m, km = self.ax, self.m, self.km
        ax.clear()
        ax.set_facecolor(TRACK_BG)
        c = self.center[k]
        x0, x1, y0, y1 = c[0] - self.half, c[0] + self.half, c[1] - self.half, c[1] + self.half

        # current field
        gx, gy = np.meshgrid(np.linspace(x0, x1, 11), np.linspace(y0, y1, 11))
        uv = self._current_at(gx.ravel())
        ax.quiver(gx.ravel() * km, gy.ravel() * km, uv[:, 0], uv[:, 1], np.hypot(uv[:, 0], uv[:, 1]), cmap=cm.cool,
                  norm=self.cnorm, alpha=0.35, scale=self.cnorm.vmax * 14, width=0.004, headwidth=4, zorder=1)

        lo = max(0, k - self.trail)
        # acoustic footprint of the ASV at the current dive depth
        reach = np.sqrt(max(m.comm_range ** 2 - m.depth[k] ** 2, 0.0))
        ax.add_patch(Circle(m.asv[k] * km, reach * km, color=ASV_COLOR, alpha=0.06, lw=0, zorder=2))
        ax.add_patch(Circle(m.asv[k] * km, reach * km, fill=False, ec=ASV_COLOR, alpha=0.45, lw=1.0, ls="--", zorder=2))
        ax.plot(m.asv[lo:k + 1, 0] * km, m.asv[lo:k + 1, 1] * km, color=ASV_COLOR, lw=1.1, alpha=0.85, zorder=3)

        for i in range(self.n):
            col, light = GLIDER_COLORS[i % 5], GLIDER_LIGHT[i % 5]
            ax.plot(m.gliders[lo:k + 1, i, 0] * km, m.gliders[lo:k + 1, i, 1] * km, color=col, lw=2.0, alpha=0.9,
                    zorder=4)
            vals, vecs = np.linalg.eigh(m.cov[k, i] * km ** 2)
            w, h = 4 * np.sqrt(np.maximum(vals[::-1], 0))
            ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
            ax.add_patch(Ellipse(m.est[k, i] * km, w, h, angle=ang, fill=False, ec=light, lw=1.3, ls="--", zorder=5))
            ax.plot(*(m.est[k, i] * km), "o", ms=7, mfc="none", mec=light, mew=1.8, zorder=5)
            if m.in_range[k, i]:
                ax.plot([m.asv[k, 0] * km, m.gliders[k, i, 0] * km], [m.asv[k, 1] * km, m.gliders[k, i, 1] * km],
                        color=col, lw=1.5, ls=":", zorder=6)

        zoom = 0.42 * 1.7 / (2 * self.half)
        if self.icons:
            self._place_icon("asv", m.asv[k] * km, np.degrees(m.heading[k]), zoom)
            for i in range(self.n):
                d = m.gliders[k, i] - m.gliders[max(k - 3, 0), i]
                self._place_icon(i, m.gliders[k, i] * km, np.degrees(np.arctan2(d[1], d[0])), zoom * 0.9)
        else:
            h = m.heading[k]
            ax.annotate("", xy=(m.asv[k] + 0.06 * np.array([np.cos(h), np.sin(h)])) * km, xytext=m.asv[k] * km,
                        arrowprops=dict(arrowstyle="-|>", color=ASV_COLOR, lw=2), zorder=8)
            ax.plot(*(m.asv[k] * km), "s", ms=7, color=ASV_COLOR, zorder=8)
            for i in range(self.n):
                ax.plot(*(m.gliders[k, i] * km), "o", ms=8, color=GLIDER_COLORS[i % 5], mec="white", mew=1.2, zorder=8)

        ax.set_xlim(x0 * km, x1 * km)
        ax.set_ylim(y0 * km, y1 * km)
        ax.set_aspect("equal")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.grid(alpha=0.2)
        if title:
            ax.set_title(title)
        if legend:
            ax.legend(handles=_legend_handles(self.n), loc="upper left", fontsize=8, ncol=2, frameon=True,
                      framealpha=0.85, edgecolor="none")


def _legend_handles(n):
    from matplotlib.lines import Line2D
    hs = [Line2D([0], [0], color=GLIDER_COLORS[i % 5], lw=2, label=f"glider {i + 1}") for i in range(n)]
    hs += [Line2D([0], [0], color=ASV_COLOR, lw=1.5, label="ASV"),
           Line2D([0], [0], ls="none", marker="o", mfc="none", mec="#999999", mew=1.5, label="EKF estimate, 2σ"),
           Line2D([0], [0], color=ASV_COLOR, ls="--", lw=1, label="acoustic range")]
    return hs


def render_mission(m: Mission, path: Path | str, title: str, subtitle: str = "", every: int = 4, fps: int = 30,
                   window_hours: float = 48.0, length_m: float = LENGTH_M, lost_fraction: float = 0.2):
    """Map, dive profile with contact timeline, and true error vs. EKF uncertainty, as an MP4."""
    km = length_m / 1000.0
    hours = 0.0375 * TIME_S / 3600.0
    speed = 2.5  # ASV top speed, m/s per nondimensional speed unit
    n = m.gliders.shape[1]
    t = np.arange(1, len(m.asv) + 1) * hours
    err, std = m.true_error * length_m, m.ekf_std * length_m
    depth_m = m.depth * length_m

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.95], height_ratios=[1, 1], left=0.06, right=0.97, top=0.86,
                              bottom=0.06, wspace=0.22, hspace=0.38)
        ax_map = fig.add_subplot(gs[:, 0])
        ax_dep = fig.add_subplot(gs[0, 1])
        ax_err = fig.add_subplot(gs[1, 1])
        mp = _Map(ax_map, m, km, speed, half_window=0.85, trail_steps=int(24 / hours))
        sm = cm.ScalarMappable(cmap=cm.cool, norm=mp.cnorm)
        cb = fig.colorbar(sm, ax=ax_map, orientation="horizontal", fraction=0.035, pad=0.09, aspect=40)
        cb.set_label("current speed (m/s)")
        fig.text(0.06, 0.95, title, fontsize=16, fontweight="bold", va="center")
        fig.text(0.06, 0.905, subtitle, fontsize=11, color="#444444", va="center")
        clock = fig.text(0.97, 0.95, "", fontsize=14, fontweight="bold", ha="right", va="center")
        status = fig.text(0.97, 0.905, "", fontsize=11, ha="right", va="center", color="#444444")
        lost_m = lost_fraction * length_m

        def draw(k):
            mp.draw(k, legend=True)
            lo = max(0.0, t[k] - window_hours)
            sel = (t >= lo) & (t <= t[k])

            ax_dep.clear()
            ax_dep.plot(t[sel], depth_m[sel], color="#3B6E8F", lw=1.6)
            ax_dep.plot(t[k], depth_m[k], "v", color="#3B6E8F", ms=8)
            band = max(depth_m.max(), 1.0)
            for i in range(n):
                y = band * (1.12 + 0.07 * i)
                ok = sel & m.in_range[:, i]
                ax_dep.scatter(t[ok], np.full(ok.sum(), y), s=4, marker="s", color=GLIDER_COLORS[i % 5], lw=0)
                ax_dep.text(lo + 0.3, y, f"G{i + 1} contact", fontsize=7, color=GLIDER_COLORS[i % 5], va="center",
                            ha="left", bbox=dict(fc=PANEL_BG, ec="none", pad=0.5))
            ax_dep.set_ylim(band * (1.18 + 0.07 * n), -band * 0.08)
            ax_dep.set_xlim(lo, max(lo + window_hours, t[k]))
            ax_dep.set_ylabel("glider depth (m)")
            ax_dep.set_title("Dives and acoustic contacts (colored ticks)", loc="left")

            ax_err.clear()
            for i in range(n):
                ax_err.plot(t[sel], err[sel, i], color=GLIDER_COLORS[i % 5], lw=1.1)
                ax_err.plot(t[sel], std[sel, i], color=GLIDER_LIGHT[i % 5], lw=1.3, ls="--")
            ax_err.axhline(lost_m, color=LOST_COLOR, lw=1, ls=":")
            ax_err.text(lo + 0.5, lost_m * 1.15, "lost (20% of separation)", color=LOST_COLOR, fontsize=8)
            ax_err.set_yscale("log")
            ax_err.set_ylim(10, max(30000, float(err[: k + 1].max()) * 1.5))
            ax_err.set_xlim(lo, max(lo + window_hours, t[k]))
            ax_err.set_xlabel("time (h)")
            ax_err.set_ylabel("position error (m)")
            ax_err.set_title("True error (solid) and EKF uncertainty (dashed)", loc="left")

            clock.set_text(f"t = {t[k]:5.1f} h")
            mean_err = err[k].mean()
            status.set_text(f"mean true error {mean_err:,.0f} m   |   in range: "
                            + " ".join(f"G{i + 1}" for i in range(n) if m.in_range[k, i]))

        writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=3000,
                              extra_args=["-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "23"])
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with writer.saving(fig, str(path), dpi=100):
            for k in range(0, len(m.asv), every):
                draw(k)
                writer.grab_frame()
        fig.savefig(path.with_suffix(".png"), dpi=60)
        plt.close(fig)
    return path


def render_comparison(missions: dict[str, Mission], path: Path | str, title: str, subtitle: str = "",
                      every: int = 4, fps: int = 30, length_m: float = LENGTH_M, lost_fraction: float = 0.2):
    """Several controllers on the same mission: a map and an error trace for each, as an MP4."""
    km = length_m / 1000.0
    hours = 0.0375 * TIME_S / 3600.0
    names = list(missions)
    T = min(len(m.asv) for m in missions.values())
    t = np.arange(1, T + 1) * hours
    lost_m = lost_fraction * length_m
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        gs = fig.add_gridspec(2, len(names), height_ratios=[1.0, 0.42], left=0.055, right=0.985, top=0.84,
                              bottom=0.08, wspace=0.28, hspace=0.32)
        axes = [fig.add_subplot(gs[0, i]) for i in range(len(names))]
        eaxes = [fig.add_subplot(gs[1, i]) for i in range(len(names))]
        maps = [_Map(ax, missions[nm], km, 2.5, half_window=1.0, trail_steps=int(24 / hours))
                for ax, nm in zip(axes, names, strict=True)]
        fig.text(0.055, 0.95, title, fontsize=16, fontweight="bold", va="center")
        fig.text(0.055, 0.905, subtitle, fontsize=11, color="#444444", va="center")
        clock = fig.text(0.985, 0.95, "", fontsize=14, fontweight="bold", ha="right", va="center")

        def draw(k):
            for i, (ax, eax, mp, nm) in enumerate(zip(axes, eaxes, maps, names, strict=True)):
                m = missions[nm]
                err = m.true_error[:T].mean(axis=1) * length_m
                lost = bool((err[: k + 1] > lost_m).any())
                mp.draw(k, legend=i == 0)
                ax.set_title(nm + ("  (lost)" if lost else ""), color=LOST_COLOR if lost else "#2E8B57")
                if i:
                    ax.set_ylabel("")
                eax.clear()
                eax.plot(t[: k + 1], err[: k + 1], color=LOST_COLOR if lost else "steelblue", lw=1.1)
                eax.axhline(lost_m, color=LOST_COLOR, lw=0.9, ls=":")
                eax.set_yscale("log")
                eax.set_ylim(30, 3e5)
                eax.set_xlim(0, t[-1])
                eax.set_xlabel("time (h)")
                if i == 0:
                    eax.set_ylabel("mean true error (m)")
                    eax.text(3, lost_m * 1.3, "lost", color=LOST_COLOR, fontsize=8)
            clock.set_text(f"t = {t[k]:5.1f} h")

        writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=3000,
                              extra_args=["-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "23"])
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with writer.saving(fig, str(path), dpi=100):
            for k in range(0, T, every):
                draw(k)
                writer.grab_frame()
        fig.savefig(path.with_suffix(".png"), dpi=60)
        plt.close(fig)
    return path
