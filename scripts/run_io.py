"""Run-folder loading for the viewers: per-image clouds (voxel-downsampled, cached)."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

from foundation_vio.io.load import stamp_ns


def load_run_config(run_dir):
    """The run's saved config.yaml, or {} if it is missing."""
    path = Path(run_dir) / "config.yaml"
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def replay_defaults(run_dir):
    """batch_size, batch_overlap, and image dir from the run's config.yaml when those paths still exist."""
    cfg = load_run_config(run_dir)
    model, files, seq = cfg.get("model", {}), cfg.get("files", {}), cfg.get("sequence", {})
    images = None
    dataset_dir, name, image_dir = files.get("dataset_dir"), seq.get("sequence_name"), files.get("image_dir")
    if dataset_dir and name and image_dir:
        path = Path(dataset_dir) / name / image_dir
        if path.is_dir():
            images = path
    return model.get("batch_size", 10), model.get("batch_overlap", 1), images


def cloud_dir(run_dir):
    """The run's per-image cloud folder; a colour-corrected *_wb_gamma copy wins when present."""
    run_dir = Path(run_dir)
    for name in ("slam_clouds_downsampled_wb_gamma", "slam_clouds_downsampled"):
        if (run_dir / name).is_dir():
            return run_dir / name
    raise FileNotFoundError(f"no slam_clouds_downsampled in {run_dir}")


def _load_one(path, voxel):
    cloud = o3d.io.read_point_cloud(str(path))
    if voxel > 0:
        cloud = cloud.voxel_down_sample(voxel)
    pts = np.asarray(cloud.points, dtype=np.float32)
    cols = (np.asarray(cloud.colors) * 255).astype(np.uint8)
    return pts, cols


def load_clouds(run_dir, voxel, cache_dir=None):
    """[(t_ns, points (n, 3) float32, colors (n, 3) uint8)] in time order, each cloud voxel-downsampled.

    The result is cached as one .npz per voxel size in <run_dir>/.viz_cache (or cache_dir).
    """
    src = cloud_dir(run_dir)
    cache = Path(cache_dir or Path(run_dir) / ".viz_cache") / f"{src.name}_voxel{voxel:g}.npz"
    if cache.exists():
        z = dict(np.load(cache))  # read each array once: NpzFile re-reads an array on every access
        bounds = np.r_[0, np.cumsum(z["counts"])]
        return [
            (int(t), z["points"][a:b], z["colors"][a:b])
            for t, a, b in zip(z["times"], bounds[:-1], bounds[1:], strict=True)
        ]

    files = sorted(src.glob("*.ply"), key=stamp_ns)
    print(f"loading {len(files)} clouds from {src} (voxel {voxel} m, cached afterwards) ...")
    with ThreadPoolExecutor(8) as pool:
        loaded = list(pool.map(lambda f: _load_one(f, voxel), files))
    clouds = [(stamp_ns(f), p, c) for f, (p, c) in zip(files, loaded, strict=True)]
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache,
            times=np.array([t for t, _, _ in clouds], dtype=np.int64),
            counts=np.array([len(p) for _, p, _ in clouds]),
            points=np.concatenate([p for _, p, _ in clouds]),
            colors=np.concatenate([c for _, _, c in clouds]),
        )
    except OSError as e:
        print(f"could not write the cache {cache}: {e}")
    return clouds


def tubes(starts, ends, radius, color, sides=8):
    """One mesh of cylinders from starts[i] to ends[i] (Open3D on macOS ignores line widths, so lines are tubes)."""
    starts, ends = np.asarray(starts, float).reshape(-1, 3), np.asarray(ends, float).reshape(-1, 3)
    d = ends - starts
    keep = np.linalg.norm(d, axis=1) > 1e-9
    starts, ends, d = starts[keep], ends[keep], d[keep]
    mesh = o3d.geometry.TriangleMesh()
    if not len(d):
        return mesh
    u = d / np.linalg.norm(d, axis=1, keepdims=True)
    ref = np.where(np.abs(u[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    v1 = np.cross(u, ref)
    v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
    v2 = np.cross(u, v1)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = radius * (np.cos(ang)[None, :, None] * v1[:, None] + np.sin(ang)[None, :, None] * v2[:, None])  # (S, k, 3)
    verts = np.concatenate([starts[:, None] + ring, ends[:, None] + ring], axis=1).reshape(-1, 3)
    k = np.arange(sides)
    quad = np.concatenate(
        [
            np.stack([k, (k + 1) % sides, sides + k], 1),
            np.stack([(k + 1) % sides, sides + (k + 1) % sides, sides + k], 1),
        ]
    )
    tris = (np.arange(len(d))[:, None, None] * 2 * sides + quad[None]).reshape(-1, 3)
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def path_tube(positions, radius, color):
    """A tube through consecutive positions."""
    positions = np.asarray(positions, float).reshape(-1, 3)
    return tubes(positions[:-1], positions[1:], radius, color)


def to_o3d(points, colors):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    return cloud


def set_view(vis, positions, front=(-0.45, -0.65, 0.62), up=(0, 0, 1), zoom=0.6):
    """An oblique global view of a z-up scene, centred on positions (the trajectory)."""
    ctr = vis.get_view_control()
    ctr.set_lookat(0.5 * (positions.min(0) + positions.max(0)))
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_zoom(zoom)
