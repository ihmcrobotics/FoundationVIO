"""One batch of model output: body poses and body-frame points of every frame, in the batch's local frame."""

import numpy as np


class Submap:
    def __init__(self, batch_id, t_ns, T_local_bodies, points, confs, colors, *, lc=False):
        self.batch_id = batch_id
        self.t_ns = t_ns  # image timestamps [ns]
        self.lc = lc  # a loop-closure image pair, not a batch
        self.T_local_bodies = np.array(T_local_bodies)  # (S, 4, 4)
        self.S, self.H, self.W, _ = points.shape
        self.points = np.asarray(points).reshape(-1, 3).astype(np.float32, copy=False)  # (S*H*W, 3), 0 where rejected
        self.confs = np.asarray(confs).reshape(-1)  # (S*H*W,), 0 where rejected
        self.colors = np.asarray(colors).reshape(-1, 3)  # (S*H*W, 3)
        self.fusion_roots = None  # (S*H*W,) duplicate grouping, see mapping/fusion.py
        self.retrieval_vectors = None  # per-frame SALAD descriptors
        self.T_world_body0 = None  # optimised pose of frame 0, set after the final solve
        self.scale = None  # model units per metre, set after the final solve
        self.cache_dir = None

    def images(self):
        return self.colors.reshape(-1, self.H, self.W, 3)

    def get_points(self, *, frame_num=None, ref_num=None):
        """(points, confs, colors) of frame_num (all if None) in ref_num's body frame (local frame if None)."""
        n = self.H * self.W
        sl = slice(None) if frame_num is None else slice((frame_num % self.S) * n, (frame_num % self.S + 1) * n)
        T = np.eye(4) if ref_num is None else np.linalg.inv(self.T_local_bodies[ref_num])
        points = (self.points[sl] @ T[:3, :3].T.astype(np.float32) + T[:3, 3].astype(np.float32)).astype(
            np.float32, copy=False
        )
        return points, self.confs[sl], self.colors[sl]

    def apply_scale(self, scale):
        """Convert poses and points from model units to metres; spilled points are converted on reload."""
        self.scale = scale
        self.T_local_bodies[:, :3, 3] /= scale
        if self.points is not None:
            self.points /= scale

    def release_points(self):
        self.points = self.confs = self.colors = self.fusion_roots = None

    def spill(self, cache_dir):
        """Write the point arrays to cache_dir and free them; reload() restores them."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / "pts.npy", self.points)
        np.save(cache_dir / "col.npy", self.colors)
        np.save(cache_dir / "conf.npy", self.confs.astype(np.float32, copy=False))
        if self.fusion_roots is not None:
            np.save(cache_dir / "roots.npy", self.fusion_roots)
        self.cache_dir = cache_dir
        self.release_points()

    def reload(self):
        self.points = np.load(self.cache_dir / "pts.npy")
        self.colors = np.load(self.cache_dir / "col.npy")
        self.confs = np.load(self.cache_dir / "conf.npy")
        roots = self.cache_dir / "roots.npy"
        self.fusion_roots = np.load(roots) if roots.exists() else None
        if self.scale is not None:  # the cache is written before scaling
            self.points /= self.scale
