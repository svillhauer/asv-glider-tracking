"""Animated GIF of an episode: top-down tracking on the left, glider dives on the right."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle, Ellipse  # noqa: E402

from .configs import LENGTH_M, TIME_S  # noqa: E402
from .rollout import Episode  # noqa: E402

ASV_COLOR = "#16365C"
GLIDER_COLORS = ["#1E88E5", "#AB47BC", "#26A69A", "#F4A261", "#E53935"]
KM = LENGTH_M / 1000.0
HOURS_PER_STEP = 0.0375 * TIME_S / 3600.0


def _ellipse(mean_km, cov, color):
    # 2-sigma position ellipse, km
    vals, vecs = np.linalg.eigh(cov * KM ** 2)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w, h = 4 * np.sqrt(np.maximum(vals[::-1], 0))
    return Ellipse(mean_km, w, h, angle=angle, fill=False, ec=color, lw=1.2, ls="--")


def render_gif(ep: Episode, path: Path | str, every: int = 16, fps: int = 12, window_km: float = 22.0,
               trail_steps: int = 288, title: str = "") -> Path:
    """Write `ep` as a GIF, one frame every `every` steps."""
    path = Path(path)
    asv, gl, est = ep.asv * KM, ep.gliders * KM, ep.estimates * KM
    depth_m = ep.depth * LENGTH_M
    hours = (np.arange(len(ep.rewards)) + 1) * HOURS_PER_STEP
    frames = np.arange(0, len(ep.rewards), every)
    colors = [GLIDER_COLORS[i % len(GLIDER_COLORS)] for i in range(ep.n_gliders)]

    fig, (ax, axd) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw=dict(width_ratios=[1.45, 1]), dpi=80)
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.13, top=0.86, wspace=0.28)

    def draw(k):
        ax.clear()
        axd.clear()
        lo = max(0, k - trail_steps)
        center = gl[k].mean(axis=0)

        # horizontal reach of the acoustic footprint at the current dive depth
        reach = np.sqrt(max(ep.comm_range ** 2 - ep.depth[k] ** 2, 0.0)) * KM
        ax.add_patch(Circle(asv[k], reach, color=ASV_COLOR, alpha=0.08, lw=0))
        ax.add_patch(Circle(asv[k], reach, fill=False, ec=ASV_COLOR, alpha=0.35, lw=1))
        ax.plot(asv[lo:k + 1, 0], asv[lo:k + 1, 1], color=ASV_COLOR, lw=1.2, alpha=0.7)

        for i, c in enumerate(colors):
            ax.plot(gl[lo:k + 1, i, 0], gl[lo:k + 1, i, 1], color=c, lw=1.5, alpha=0.45)
            ax.plot(*gl[k, i], "o", ms=8, color=c, mec="white", mew=1.2, zorder=4)
            ax.plot(*est[k, i], "x", ms=7, color=c, mew=2, zorder=4)
            ax.annotate(f"G{i + 1}", gl[k, i], xytext=(7, 7), textcoords="offset points", fontsize=8, color=c)
            ax.add_patch(_ellipse(est[k, i], ep.covariances[k, i], c))
            if ep.in_range[k, i]:
                ax.plot([asv[k, 0], gl[k, i, 0]], [asv[k, 1], gl[k, i, 1]], color=c, lw=1.4, ls=":", zorder=3)

        # ASV as an arrow along its heading
        h = ep.heading[k]
        ax.annotate("", xy=asv[k] + 2.2 * np.array([np.cos(h), np.sin(h)]), xytext=asv[k],
                    arrowprops=dict(arrowstyle="-|>", color=ASV_COLOR, lw=2), zorder=5)
        ax.plot(*asv[k], "s", ms=7, color=ASV_COLOR, zorder=5)

        ax.set_xlim(center[0] - window_km, center[0] + window_km)
        ax.set_ylim(center[1] - window_km, center[1] + window_km)
        ax.set_aspect("equal")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.set_title("Top-down: true (o), EKF estimate (x), 2σ ellipse", fontsize=10)
        ax.grid(alpha=0.2)

        # dive profile over the last two cycles
        span = slice(max(0, k - 240), k + 1)
        axd.fill_between(hours[span], depth_m[span], 0, color="#90CAF9", alpha=0.25, lw=0)
        axd.plot(hours[span], depth_m[span], color="#1565C0", lw=1.8)
        for i, c in enumerate(colors):
            off = (i - (ep.n_gliders - 1) / 2) * 0.35
            axd.plot(hours[k] + off, depth_m[k], "o", ms=8, color=c, mec="white", mew=1.2, zorder=4)
        axd.set_xlim(hours[span][0], hours[span][0] + 240 * HOURS_PER_STEP)
        axd.set_ylim(ep.max_depth * LENGTH_M * 1.12, -120)
        axd.set_xlabel("time (h)")
        axd.set_ylabel("glider depth (m)")
        axd.set_title("Synchronized sawtooth dives", fontsize=10)
        axd.grid(alpha=0.2)

        n_contact = int(ep.in_range[k].sum())
        fig.suptitle(f"{title}   t = {hours[k]:5.1f} h   contacts: {n_contact}/{ep.n_gliders}", fontsize=11)

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path
