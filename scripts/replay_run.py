"""Replay a finished run as if live: each batch's clouds appear when the batch completes, the trajectory grows.

pixi run replay <run_dir> [--images <image dir>] [--speed 2] [--ceiling 0.7]
Keys (either window): space pause/play, N next batch, R restart; Q or Esc quits the 3D window. --paused starts paused.
--record out.mp4 renders the whole replay offscreen instead, with the camera image inset top right.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from run_io import load_clouds, path_tube, replay_defaults, set_view, to_o3d

from foundation_vio.io.load import stamp_ns
from foundation_vio.io.results import load_tum

SLAM_COLOR = (1.0, 0.45, 0.0)


def build_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--voxel", type=float, default=0.01, help="voxel size [m] for each cloud (default 0.01)")
    p.add_argument("--batch", type=int, help="images per batch (default: the run's model.batch_size)")
    p.add_argument("--overlap", type=int, help="images shared by consecutive batches (default: model.batch_overlap)")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed, 1 = real time")
    p.add_argument(
        "--line-radius", type=float, help="trajectory tube radius [m] (default 0.06 %% of the map size, 1.5 cm min)"
    )
    p.add_argument(
        "--images", type=Path, help="image folder for the camera window (default: the run's files.image_dir)"
    )
    p.add_argument("--zmax", type=float, help="hide points above this height [m]")
    p.add_argument("--ceiling", type=float, help="hide points more than this [m] above the highest pose (ceiling cut)")
    p.add_argument("--paused", action="store_true", help="start paused (space plays)")
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--dark", action="store_true", help="dark background")
    p.add_argument("--size", default="1600x1000", help="window size WxH")
    p.add_argument("--record", type=Path, help="write the replay to this MP4 instead of opening a window")
    p.add_argument("--fps", type=int, default=30, help="video frame rate for --record")
    return p.parse_args()


def reveal_times(t_pose, batch, overlap):
    """Time at which each pose is known: the time of the last image of the first batch that contains it."""
    n, stride = len(t_pose), batch - overlap
    j = np.arange(n)
    first_batch = np.maximum(0, np.ceil((j - batch + 1) / stride)).astype(int)
    return t_pose[np.minimum(first_batch * stride + batch - 1, n - 1)]


def batch_clouds(clouds, t_pose, t_reveal, zmax, voxel):
    """[(reveal time, Open3D cloud)] with one merged cloud per batch."""
    groups = {}
    for t_ns, pts, cols in clouds:
        j = min(np.searchsorted(t_pose, t_ns * 1e-9 - 1e-6), len(t_pose) - 1)
        keep = pts[:, 2] <= zmax
        groups.setdefault(t_reveal[j], []).append((pts[keep], cols[keep]))
    out = []
    for t in sorted(groups):
        pts = np.concatenate([p for p, _ in groups[t]])
        cols = np.concatenate([c for _, c in groups[t]])
        out.append((t, to_o3d(pts, cols).voxel_down_sample(voxel)))
    return out


def set_mesh(dst, src):
    """Copy src's geometry into dst in place, so the visualizer can update dst."""
    dst.vertices, dst.triangles = src.vertices, src.triangles
    dst.vertex_normals, dst.vertex_colors = src.vertex_normals, src.vertex_colors


class Camera:
    """The camera window: the image nearest the replay time."""

    def __init__(self, folder, height=400):
        self.files = sorted([*folder.glob("*.jpg"), *folder.glob("*.png")], key=stamp_ns)
        self.t = np.array([stamp_ns(f) for f in self.files]) * 1e-9
        self.height, self.i, self.img = height, -1, None

    def image(self, t):
        """BGR image nearest t, resized to self.height."""
        i = int(np.clip(np.searchsorted(self.t, t), 0, len(self.files) - 1))
        if i != self.i:
            img = cv2.imread(str(self.files[i]))
            self.i, self.img = i, cv2.resize(img, (int(img.shape[1] * self.height / img.shape[0]), self.height))
        return self.img

    def show(self, t):
        cv2.imshow("camera", self.image(t))
        return cv2.waitKey(1)  # the camera window takes focus, so it reads the keys too


def main():
    a = build_args()
    batch, overlap, images = replay_defaults(a.run_dir)
    a.batch = a.batch if a.batch is not None else batch
    a.overlap = a.overlap if a.overlap is not None else overlap
    a.images = a.images or images
    t_pose, T_slam = load_tum(a.run_dir / "tum_poses.csv")
    t_reveal = reveal_times(t_pose, a.batch, a.overlap)
    zmax = min(
        a.zmax if a.zmax is not None else np.inf,
        T_slam[:, 2, 3].max() + (a.ceiling if a.ceiling is not None else np.inf),
    )
    batches = batch_clouds(load_clouds(a.run_dir, a.voxel), t_pose, t_reveal, zmax, a.voxel)
    w, h = (int(v) for v in a.size.split("x"))
    camera = Camera(a.images, h // 3 if a.record else 400) if a.images else None
    print(f"{len(batches)} batches over {t_pose[-1] - t_pose[0]:.0f} s")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"foundation_vio replay: {a.run_dir.name}",
        width=w,
        height=h,
        visible=not a.record,
    )
    opt = vis.get_render_option()
    opt.point_size = a.point_size
    bg = np.full(3, 0.07 if a.dark else 1.0)
    opt.background_color = bg
    # two background-coloured points span the final map (2-98 %, ignoring strays), so the view is fixed from the start
    pts = np.concatenate([np.asarray(c.points)[::10] for _, c in batches] + [T_slam[:, :3, 3]])
    extent = np.percentile(pts, [2, 98], axis=0)
    frame = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(extent))
    a.line_radius = a.line_radius or max(0.015, 0.0006 * np.linalg.norm(extent[1] - extent[0]))
    frame.paint_uniform_color(bg)
    vis.add_geometry(frame)
    tracks = {"slam": o3d.geometry.TriangleMesh()}
    for mesh in tracks.values():
        vis.add_geometry(mesh, reset_bounding_box=False)
    set_view(vis, extent)

    state = {"t": t_pose[0], "playing": not a.paused, "next": 0, "shown": 0, "last": time.perf_counter()}

    def restart(_vis):
        for _, cloud in batches[: state["next"]]:
            vis.remove_geometry(cloud, reset_bounding_box=False)
        for mesh in tracks.values():
            set_mesh(mesh, o3d.geometry.TriangleMesh())
            vis.update_geometry(mesh)
        state.update(t=t_pose[0], next=0, shown=0)

    def toggle(_vis):
        state["playing"] = not state["playing"]
        print("playing" if state["playing"] else f"paused at {state['t'] - t_pose[0]:.1f} s")

    def step(_vis):
        if state["next"] < len(batches):
            state["t"] = batches[state["next"]][0]

    keys = {ord(" "): toggle, ord("N"): step, ord("R"): restart}
    for k, f in keys.items():
        vis.register_key_callback(k, f)

    def update():
        while state["next"] < len(batches) and batches[state["next"]][0] <= state["t"]:
            vis.add_geometry(batches[state["next"]][1], reset_bounding_box=False)
            state["next"] += 1
        n = int(np.searchsorted(t_reveal, state["t"], side="right"))  # poses known so far
        if n != state["shown"] and n > 0:
            set_mesh(tracks["slam"], path_tube(T_slam[:n, :3, 3], a.line_radius, SLAM_COLOR))
            for mesh in tracks.values():
                vis.update_geometry(mesh)
            state["shown"] = n
        if camera and not a.record:
            f = keys.get(ord(chr(camera.show(state["t"]) & 0xFF).upper()))
            if f:
                f(vis)

    if a.record:
        record(vis, update, state, camera, t_pose[-1] + 2 * a.speed, a)  # hold the final map for 2 s
        vis.destroy_window()
        return

    while vis.poll_events():
        now = time.perf_counter()
        if state["playing"] and state["t"] < t_pose[-1]:
            state["t"] = min(t_pose[-1], state["t"] + (now - state["last"]) * a.speed)
        state["last"] = now
        update()
        vis.update_renderer()
    vis.destroy_window()
    cv2.destroyAllWindows()


def record(vis, update, state, camera, t_end, a):
    """Step the replay clock by speed / fps per frame and write each rendered frame to a.record."""
    a.record.parent.mkdir(parents=True, exist_ok=True)
    writer, n = None, 0
    while state["t"] <= t_end:
        update()
        vis.poll_events()
        vis.update_renderer()
        frame = (np.asarray(vis.capture_screen_float_buffer(do_render=True))[..., ::-1] * 255).astype(np.uint8)
        if camera:
            img = camera.image(state["t"])
            m = 16  # margin [px]
            frame[m : m + img.shape[0], -m - img.shape[1] : -m] = img
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(a.record), fourcc, a.fps, (frame.shape[1], frame.shape[0]))
        writer.write(np.ascontiguousarray(frame))
        state["t"] += a.speed / a.fps
        n += 1
    writer.release()
    print(f"wrote {a.record}: {n} frames, {n / a.fps:.0f} s")


if __name__ == "__main__":
    main()
