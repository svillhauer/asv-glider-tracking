"""Animated GIF of an episode: top-down tracking on the left, the mission in 3D on the right."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle, Ellipse  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from .configs import LENGTH_M, TIME_S  # noqa: E402
from .rollout import Episode  # noqa: E402

ASV_COLOR = "#16365C"
GLIDER_COLORS = ["#1E88E5", "#AB47BC", "#26A69A", "#F4A261", "#E53935"]
PANE = (0.941, 0.957, 0.973, 1.0)
KM = LENGTH_M / 1000.0
HOURS_PER_STEP = 0.0375 * TIME_S / 3600.0


def _ellipse(mean_km, cov, color):
    # 2-sigma position ellipse, km
    vals, vecs = np.linalg.eigh(cov * KM ** 2)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w, h = 4 * np.sqrt(np.maximum(vals[::-1], 0))
    return Ellipse(mean_km, w, h, angle=angle, fill=False, ec=color, lw=1.2, ls="--")


def _draw_top_down(ax, ep, k, asv, gl, est, colors, window_km, trail_steps):
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


def _clip(xy, z, center, half):
    # hide points outside the moving window so lines end at its edges
    out = (np.abs(xy[:, 0] - center[0]) > half) | (np.abs(xy[:, 1] - center[1]) > half)
    xy, z = xy.astype(float).copy(), z.astype(float).copy()
    xy[out], z[out] = np.nan, np.nan
    return xy[:, 0], xy[:, 1], z


def _draw_3d(ax, ep, k, asv, gl, depth_m, colors, half_km, history_steps, azim):
    center = gl[k].mean(axis=0)
    x0, x1, y0, y1 = center[0] - half_km, center[0] + half_km, center[1] - half_km, center[1] + half_km
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(PANE)
        axis.pane.set_edgecolor("white")

    # sea surface
    xx, yy = np.meshgrid([x0, x1], [y0, y1])
    ax.plot_surface(xx, yy, np.zeros_like(xx), color="skyblue", alpha=0.12, shade=False)

    lo = max(0, k - history_steps)
    for i, c in enumerate(colors):
        ax.plot(*_clip(gl[lo:k + 1, i], -depth_m[lo:k + 1], center, half_km), color=c, lw=1.5, alpha=0.85)
        ax.scatter(gl[k, i, 0], gl[k, i, 1], -depth_m[k], s=36, color=c, edgecolors="white", linewidths=0.8,
                   depthshade=False)
        if ep.in_range[k, i]:
            ax.plot([asv[k, 0], gl[k, i, 0]], [asv[k, 1], gl[k, i, 1]], [0, -depth_m[k]], color=c, lw=1.5, ls=":")
    ax.plot(*_clip(asv[lo:k + 1], np.zeros(k + 1 - lo), center, half_km), color=ASV_COLOR, lw=1.0, alpha=0.8)
    ax.scatter(asv[k, 0], asv[k, 1], 0, s=46, marker="s", color=ASV_COLOR, depthshade=False)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(-ep.max_depth * LENGTH_M * 1.1, 0)
    ticks = np.linspace(0, ep.max_depth * LENGTH_M, 4)
    ax.set_zticks(-ticks)
    ax.set_zticklabels([f"{t:.0f}" for t in ticks])
    ax.set_box_aspect((1.0, 1.0, 0.55), zoom=1.2)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_xlabel("x (km)", labelpad=2)
    ax.set_ylabel("y (km)", labelpad=2)
    ax.set_zlabel("depth (m)", labelpad=6)
    ax.tick_params(labelsize=7, pad=0)
    ax.view_init(elev=24, azim=azim)
    ax.set_title("3D view, last 2 days (depth exaggerated)", fontsize=10)


def render_gif(ep: Episode, path: Path | str, every: int = 16, fps: int = 12, window_km: float = 22.0,
               trail_steps: int = 288, window_3d_km: float = 30.0, history_3d_steps: int = 576,
               title: str = "") -> Path:
    """Write `ep` as a GIF, one frame every `every` steps."""
    path = Path(path)
    asv, gl, est = ep.asv * KM, ep.gliders * KM, ep.estimates * KM
    depth_m = ep.depth * LENGTH_M
    hours = (np.arange(len(ep.rewards)) + 1) * HOURS_PER_STEP
    frames = np.arange(0, len(ep.rewards), every)
    colors = [GLIDER_COLORS[i % len(GLIDER_COLORS)] for i in range(ep.n_gliders)]

    fig = plt.figure(figsize=(12.5, 5.2), dpi=80)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45], left=0.055, right=0.995, bottom=0.11, top=0.88, wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[0, 1], projection="3d")

    def draw(k):
        ax.clear()
        ax3.clear()
        _draw_top_down(ax, ep, k, asv, gl, est, colors, window_km, trail_steps)
        # slow camera sweep over the mission
        _draw_3d(ax3, ep, k, asv, gl, depth_m, colors, window_3d_km, history_3d_steps,
                 azim=-65 + 40 * k / len(ep.rewards))
        n_contact = int(ep.in_range[k].sum())
        fig.suptitle(f"{title}   t = {hours[k]:5.1f} h   contacts: {n_contact}/{ep.n_gliders}", fontsize=11)

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path
