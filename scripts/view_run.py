"""Open a finished run in an Open3D window: the merged map and the run's trajectory.

pixi run view <run_dir> [--ceiling 0.7]
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from run_io import load_clouds, path_tube, set_view, to_o3d

from foundation_vio.io.results import load_tum

TRAJ_COLOR = (1.0, 0.45, 0.0)


def build_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--voxel", type=float, default=0.01, help="voxel size [m] for the map (default 0.01)")
    p.add_argument("--line-radius", type=float, default=0.015, help="trajectory tube radius [m]")
    p.add_argument("--zmax", type=float, help="hide points above this height [m]")
    p.add_argument("--ceiling", type=float, help="hide points more than this [m] above the highest pose (ceiling cut)")
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--dark", action="store_true", help="dark background")
    p.add_argument("--size", default="1600x1000", help="window size WxH")
    p.add_argument("--screenshot", type=Path, help="render once to this PNG and exit")
    return p.parse_args()


def main():
    a = build_args()
    clouds = load_clouds(a.run_dir, a.voxel)
    _, T_slam = load_tum(a.run_dir / "tum_poses.csv")
    points = np.concatenate([p for _, p, _ in clouds])
    colors = np.concatenate([c for _, _, c in clouds])
    zmax = min(
        a.zmax if a.zmax is not None else np.inf,
        T_slam[:, 2, 3].max() + (a.ceiling if a.ceiling is not None else np.inf),
    )
    if np.isfinite(zmax):
        keep = points[:, 2] <= zmax
        points, colors = points[keep], colors[keep]
    cloud = to_o3d(points, colors).voxel_down_sample(a.voxel)  # drop duplicates where frames overlap
    print(f"{len(clouds)} clouds -> {len(cloud.points):,} points")

    r = a.line_radius
    geoms = [cloud, path_tube(T_slam[:, :3, 3], r, TRAJ_COLOR)]

    w, h = (int(v) for v in a.size.split("x"))
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"foundation_vio: {a.run_dir.name}", width=w, height=h, visible=a.screenshot is None)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = a.point_size
    opt.background_color = np.full(3, 0.07 if a.dark else 1.0)
    set_view(vis, T_slam[:, :3, 3])  # framed on the trajectory: stray far points do not shift the view
    if a.screenshot:
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(a.screenshot), do_render=True)
        print(f"saved {a.screenshot}")
    else:
        vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
