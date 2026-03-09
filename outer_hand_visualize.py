import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.config import STGCN_PATHS_PATH
from src.stgcn.stgcn_dataset import (
    infer_outer_hand_side,
    PARTITION_HIP_INDICES,
)


EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
    [9, 10], [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21],
    [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [11, 23], [12, 24],
    [23, 24], [23, 25], [25, 27], [27, 29], [27, 31], [24, 26], [26, 28],
    [28, 30], [28, 32],
]

LEFT_ARM_JOINTS = {11, 13, 15, 17, 19, 21}
RIGHT_ARM_JOINTS = {12, 14, 16, 18, 20, 22}


def _set_axes_equal(ax, points):
    if points.size == 0:
        return
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    if max_range == 0:
        max_range = 1.0
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def _filter_arm_edges(edges, joint_set):
    return [e for e in edges if e[0] in joint_set and e[1] in joint_set]


def _gather_paths(input_path, paths_file, num_samples, seed):
    paths = []

    if input_path:
        if os.path.isdir(input_path):
            for root, _, files in os.walk(input_path):
                for name in files:
                    if name.endswith(".npy") and "_subspace_features" not in name:
                        paths.append(os.path.join(root, name))
        else:
            paths.append(input_path)

    if not paths:
        if paths_file is None:
            paths_file = STGCN_PATHS_PATH
        if os.path.exists(paths_file):
            loaded = np.load(paths_file, allow_pickle=True)
            paths.extend([str(p) for p in loaded.tolist()])

    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise FileNotFoundError("No valid .npy paths found.")

    paths = sorted(set(paths))
    if num_samples is not None and num_samples > 0 and len(paths) > num_samples:
        rng = np.random.default_rng(seed)
        paths = rng.choice(paths, size=num_samples, replace=False).tolist()
    return paths


def _resolve_frame_index(frame_arg, num_frames):
    if frame_arg is None:
        return num_frames // 2
    if frame_arg < 0:
        return max(0, num_frames + frame_arg)
    return min(frame_arg, num_frames - 1)


def _plot_skeleton(ax, skeleton_frame, base_color="#9AA0A6", base_alpha=0.5):
    x = skeleton_frame[:, 0]
    y = skeleton_frame[:, 1]
    z = skeleton_frame[:, 2]
    ax.scatter(x, y, z, c=base_color, marker="o", s=12, alpha=base_alpha, depthshade=False)
    for edge in EDGES:
        p1 = skeleton_frame[edge[0]]
        p2 = skeleton_frame[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=base_color, alpha=base_alpha, linewidth=1)


def _plot_arm(ax, skeleton_frame, joint_set, color, label, linewidth=2.5, point_size=38):
    edges = _filter_arm_edges(EDGES, joint_set)
    for edge in edges:
        p1 = skeleton_frame[edge[0]]
        p2 = skeleton_frame[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=linewidth)
    pts = skeleton_frame[list(joint_set)]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=point_size, label=label, depthshade=False)


def _rotate_x(data, deg):
    if deg == 0:
        return data
    rad = np.deg2rad(deg)
    c = np.cos(rad)
    s = np.sin(rad)
    rot = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)
    flat = data.reshape(-1, 3)
    rotated = flat @ rot.T
    return rotated.reshape(data.shape)


def _apply_display_transform(data, rotate_x_deg=0.0):
    if rotate_x_deg == 0:
        return data
    return _rotate_x(data, rotate_x_deg)


def visualize_one(path, out_dir, frame_arg, axis, invert, display_rotate_x=-90.0, show=False, view=(20, -70)):
    data = np.load(path)
    if data.ndim != 3 or data.shape[1] != 33 or data.shape[2] != 3:
        print(f"Skip {path}: expected shape (T, 33, 3), got {data.shape}")
        return

    side, dx, start, end = infer_outer_hand_side(
        data,
        axis=axis,
        invert=invert,
        return_delta=True,
    )

    display_data = _apply_display_transform(data, rotate_x_deg=display_rotate_x)
    frame_idx = _resolve_frame_index(frame_arg, display_data.shape[0])
    skeleton_frame = display_data[frame_idx]

    outer_set = RIGHT_ARM_JOINTS if side == "right" else LEFT_ARM_JOINTS
    inner_set = LEFT_ARM_JOINTS if side == "right" else RIGHT_ARM_JOINTS

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    _plot_skeleton(ax, skeleton_frame)
    _plot_arm(ax, skeleton_frame, inner_set, color="#6C8EBF", label="inner hand", linewidth=2.0, point_size=30)
    _plot_arm(ax, skeleton_frame, outer_set, color="#E74C3C", label="outer hand", linewidth=3.0, point_size=50)

    hip_center = np.nanmean(display_data[:, PARTITION_HIP_INDICES, :], axis=1)
    if hip_center.size > 0 and np.all(np.isfinite(hip_center)):
        ax.plot(
            hip_center[:, 0],
            hip_center[:, 1],
            hip_center[:, 2],
            color="#444444",
            alpha=0.35,
            linewidth=1.5,
            label="hip center path",
        )
        ax.scatter(hip_center[0, 0], hip_center[0, 1], hip_center[0, 2], c="#2ECC71", s=36, label="start")
        ax.scatter(hip_center[-1, 0], hip_center[-1, 1], hip_center[-1, 2], c="#F1C40F", s=36, label="end")

    title = (
        f"{os.path.basename(path)} | frame {frame_idx} | "
        f"outer={side} | axis={axis} | dx={dx:.4f}"
    )
    if invert:
        title += " | invert"
    if display_rotate_x != 0:
        title += f" | rotX={display_rotate_x:g}"
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=view[0], azim=view[1])

    _set_axes_equal(ax, skeleton_frame)
    ax.legend(loc="upper left", fontsize=8)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    suffix = f"outer-{side}_frame{frame_idx}_axis{axis}"
    if invert:
        suffix += "_inv"
    out_path = os.path.join(out_dir, f"{base}_{suffix}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved: {out_path}")

    return {
        "side": side,
        "dx": dx,
        "data": data,
        "display_data": display_data,
        "base": base,
        "suffix": suffix,
    }


def _set_axes_from_sequence(ax, data):
    flat = data.reshape(-1, 3)
    _set_axes_equal(ax, flat)


def create_gif(display_data, side, axis, invert, out_dir, base, suffix, fps=12, stride=2, max_frames=200, view=(20, -70), display_rotate_x=-90.0):
    indices = list(range(0, display_data.shape[0], max(1, stride)))
    if max_frames is not None and max_frames > 0 and len(indices) > max_frames:
        indices = indices[:max_frames]

    outer_set = RIGHT_ARM_JOINTS if side == "right" else LEFT_ARM_JOINTS
    inner_set = LEFT_ARM_JOINTS if side == "right" else RIGHT_ARM_JOINTS

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    base_lines = []
    for _ in EDGES:
        line, = ax.plot([], [], [], color="#9AA0A6", alpha=0.5, linewidth=1)
        base_lines.append(line)

    outer_edges = _filter_arm_edges(EDGES, outer_set)
    inner_edges = _filter_arm_edges(EDGES, inner_set)
    outer_lines = []
    inner_lines = []
    for _ in outer_edges:
        line, = ax.plot([], [], [], color="#E74C3C", linewidth=3.0)
        outer_lines.append(line)
    for _ in inner_edges:
        line, = ax.plot([], [], [], color="#6C8EBF", linewidth=2.0)
        inner_lines.append(line)

    base_scatter = ax.scatter([], [], [], c="#9AA0A6", s=12, alpha=0.5, depthshade=False)
    outer_scatter = ax.scatter([], [], [], c="#E74C3C", s=50, depthshade=False)
    inner_scatter = ax.scatter([], [], [], c="#6C8EBF", s=30, depthshade=False)

    hip_center = np.nanmean(display_data[:, PARTITION_HIP_INDICES, :], axis=1)
    if hip_center.size > 0 and np.all(np.isfinite(hip_center)):
        ax.plot(
            hip_center[:, 0],
            hip_center[:, 1],
            hip_center[:, 2],
            color="#444444",
            alpha=0.35,
            linewidth=1.5,
            label="hip center path",
        )
        ax.scatter(hip_center[0, 0], hip_center[0, 1], hip_center[0, 2], c="#2ECC71", s=36, label="start")
        ax.scatter(hip_center[-1, 0], hip_center[-1, 1], hip_center[-1, 2], c="#F1C40F", s=36, label="end")

    title = f"{base} | outer={side} | axis={axis}"
    if invert:
        title += " | invert"
    if display_rotate_x != 0:
        title += f" | rotX={display_rotate_x:g}"
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=view[0], azim=view[1])
    _set_axes_from_sequence(ax, display_data)
    ax.legend(loc="upper left", fontsize=8)

    text = ax.text2D(0.02, 0.02, "", transform=ax.transAxes, fontsize=9)

    def _update(frame_idx):
        frame = display_data[frame_idx]
        for line, edge in zip(base_lines, EDGES):
            p1 = frame[edge[0]]
            p2 = frame[edge[1]]
            line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            line.set_3d_properties([p1[2], p2[2]])

        for line, edge in zip(outer_lines, outer_edges):
            p1 = frame[edge[0]]
            p2 = frame[edge[1]]
            line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            line.set_3d_properties([p1[2], p2[2]])

        for line, edge in zip(inner_lines, inner_edges):
            p1 = frame[edge[0]]
            p2 = frame[edge[1]]
            line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            line.set_3d_properties([p1[2], p2[2]])

        base_scatter._offsets3d = (frame[:, 0], frame[:, 1], frame[:, 2])
        outer_pts = frame[list(outer_set)]
        inner_pts = frame[list(inner_set)]
        outer_scatter._offsets3d = (outer_pts[:, 0], outer_pts[:, 1], outer_pts[:, 2])
        inner_scatter._offsets3d = (inner_pts[:, 0], inner_pts[:, 1], inner_pts[:, 2])
        text.set_text(f"frame {frame_idx}")

        artists = base_lines + outer_lines + inner_lines + [base_scatter, outer_scatter, inner_scatter, text]
        return artists

    animation = FuncAnimation(fig, _update, frames=indices, interval=1000 / max(1, fps), blit=False)

    os.makedirs(out_dir, exist_ok=True)
    gif_path = os.path.join(out_dir, f"{base}_{suffix}.gif")
    try:
        animation.save(gif_path, writer="pillow", fps=fps)
        print(f"Saved: {gif_path}")
    except Exception as exc:
        print(f"Failed to save GIF for {base}: {exc}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize outer-hand selection on skeleton data.")
    parser.add_argument("--input", type=str, default=None, help="Single .npy file or directory of .npy files")
    parser.add_argument("--paths_file", type=str, default=None, help="Path to stgcn_paths.npy")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame", type=int, default=None, help="Frame index (default: middle). Use -1 for last frame.")
    parser.add_argument("--axis", type=str, default="x", choices=["x", "y", "z"])
    parser.add_argument("--invert", action="store_true", help="Invert outer-hand decision")
    parser.add_argument("--out_dir", type=str, default="results/outer_hand_vis")
    parser.add_argument("--view", type=str, default="20,-70", help="Camera view as 'elev,azim'")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--display_rotate_x", type=float, default=-90.0, help="Rotate display around X axis in degrees (use 0 to disable)")
    parser.add_argument("--no_gif", action="store_true", help="Disable GIF output")
    parser.add_argument("--gif_fps", type=int, default=12, help="GIF frames per second")
    parser.add_argument("--gif_stride", type=int, default=2, help="Take every N frames for GIF")
    parser.add_argument("--gif_max_frames", type=int, default=200, help="Max frames in GIF (0 for no limit)")

    args = parser.parse_args()
    view_parts = [p.strip() for p in args.view.split(",")]
    if len(view_parts) != 2:
        raise ValueError("Invalid --view format. Use 'elev,azim' (e.g., 20,-70)")
    view = (float(view_parts[0]), float(view_parts[1]))

    paths = _gather_paths(args.input, args.paths_file, args.num_samples, args.seed)
    for path in paths:
        info = visualize_one(
            path=path,
            out_dir=args.out_dir,
            frame_arg=args.frame,
            axis=args.axis,
            invert=args.invert,
            display_rotate_x=args.display_rotate_x,
            show=args.show,
            view=view,
        )
        if info and not args.no_gif:
            max_frames = None if args.gif_max_frames == 0 else args.gif_max_frames
            create_gif(
                display_data=info["display_data"],
                side=info["side"],
                axis=args.axis,
                invert=args.invert,
                out_dir=args.out_dir,
                base=info["base"],
                suffix=info["suffix"],
                fps=args.gif_fps,
                stride=args.gif_stride,
                max_frames=max_frames,
                view=view,
                display_rotate_x=args.display_rotate_x,
            )


if __name__ == "__main__":
    main()
