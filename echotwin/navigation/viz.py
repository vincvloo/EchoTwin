"""Plots and animation."""
from __future__ import annotations

import io

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import numpy as np              
from matplotlib.patches import Wedge

TEAL, AMBER, INK = "#0b6e73", "#c27a00", "#16201f"


def _map_rgb(gmap, world=None):
    img = np.full(gmap.occ.shape + (3,), 0.78)
    img[gmap.known & ~gmap.occ] = 0.98
    if world is not None:
        img[world.occ & ~gmap.occ] = (0.85, 0.35, 0.25)    # real surface where the map has none
    img[gmap.occ] = 0.12
    return img


def plot_maps(gmap, world, path, differs="glass, moved chair, 1.5 % scale drift"):
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=110)
    ax.imshow(_map_rgb(gmap, world), origin="lower", extent=gmap.extent, interpolation="nearest")
    ax.set_title("Map built from the phone scan (black)"
                 + (f"\nred = where reality differs ({differs})" if differs else "\nsimulated world = map"), fontsize=10)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _draw_objects(ax, objects):
    """Dashed outline + class name for each object from echotwin.perception.objects."""
    for o in objects:
        ax.add_patch(plt.Rectangle((o["x"] - o["size_x"] / 2, o["y"] - o["size_y"] / 2), o["size_x"], o["size_y"],
                                   fill=False, ec=INK, lw=0.8, ls="--", alpha=0.7))
        ax.text(o["x"], o["y"], o["class"], ha="center", va="center", fontsize=7, color=INK,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))


def animate(gmap, world, rig, ep, path, fps=10, objects=None):
    frames = []
    true, est = np.array(ep.true), np.array(ep.est)
    pos_err, _ = ep.errors()
    base = _map_rgb(gmap, world)
    for k, parts in ep.snapshots:
        fig, ax = plt.subplots(figsize=(6.4, 5), dpi=80)
        ax.imshow(base, origin="lower", extent=gmap.extent, interpolation="nearest")
        if objects:
            _draw_objects(ax, objects)
        ax.scatter(parts[:, 0], parts[:, 1], s=1, c=TEAL, alpha=0.35, linewidths=0)
        x, y, th = true[k]
        for (dx, dy, dth), z in zip(rig.mounts, ep.z[k]):
            sx = x + np.cos(th) * dx - np.sin(th) * dy
            sy = y + np.sin(th) * dx + np.cos(th) * dy
            a = np.degrees(th + dth)
            ax.add_patch(Wedge((sx, sy), z, a - np.degrees(rig.fov) / 2, a + np.degrees(rig.fov) / 2,
                               color=AMBER, alpha=0.25, lw=0))
        ax.plot(true[: k + 1, 0], true[: k + 1, 1], color=INK, lw=0.8, alpha=0.5)
        ax.add_patch(plt.Circle((x, y), 0.12, color=INK))
        ax.plot([x, x + 0.25 * np.cos(th)], [y, y + 0.25 * np.sin(th)], color="white", lw=1.5)
        ex, ey, eth = est[k]
        ax.add_patch(plt.Circle((ex, ey), 0.14, fill=False, color=TEAL, lw=2))
        ax.plot([ex, ex + 0.3 * np.cos(eth)], [ey, ey + 0.3 * np.sin(eth)], color=TEAL, lw=2)
        ax.set_xlim(gmap.extent[:2]); ax.set_ylim(gmap.extent[2:]); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t = {ep.t[k]:5.1f} s   particles = {len(parts):4d}   error = {pos_err[k]*100:5.0f} cm",
                     fontsize=9, family="monospace")
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
        frames.append(imageio.imread(buf.getvalue()))
    imageio.mimsave(path, frames, duration=1 / fps, loop=0)


def plot_errors(ep, path, kidnap_at=None):
    pos, ang = ep.errors()
    t = np.array(ep.t)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 4.5), dpi=110, sharex=True)
    a1.plot(t, pos * 100, color=TEAL); a1.axhline(30, ls="--", color="grey", lw=0.8)
    a1.set_ylabel("position error (cm)"); a1.set_yscale("log")
    a2.plot(t, np.degrees(ang), color=AMBER); a2.set_ylabel("heading error (deg)"); a2.set_xlabel("time (s)")
    if kidnap_at is not None:
        for a in (a1, a2):
            a.axvline(t[kidnap_at], color="red", lw=1)
        a1.text(t[kidnap_at], a1.get_ylim()[1] * 0.6, "  kidnapped", color="red", fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def write_benchmark(rows, path):
    lines = ["| Configuration | Converged | Distance to converge (median) | Time (median) | Position error after | Heading error after |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['config']} | {r['converged']}/{r['runs']} | {r['median_dist_m']:.1f} m | "
                     f"{r['median_time_s']:.0f} s | {r['pos_err_cm']:.0f} cm | {r['heading_err_deg']:.1f} deg |")
    path.write_text("\n".join(lines) + "\n")
