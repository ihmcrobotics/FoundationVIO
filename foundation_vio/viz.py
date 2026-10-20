"""Live Rerun viewer: one cloud per batch from the live iSAM; relocate once after optimize()."""

import logging

import numpy as np
from gtsam.symbol_shorthand import S, X

from foundation_vio.graph.graph import Graph
from foundation_vio.math.matrix import get_base_idx, transform_cloud_by_pose

log = logging.getLogger(__name__)
SLAM_COLOR = (255, 115, 0)
VOXEL = 0.01


class LiveViewer:
    def __init__(self, ceiling=None):
        self.ceiling = ceiling
        self.enabled = False
        self._chunks = []  # (batch_id, pts_model, colors 0-1)
        try:
            import rerun as rr  # noqa: PLC0415

            rr.init("foundation_vio", spawn=True)
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        except ImportError:
            log.warning("--view needs rerun-sdk; pixi install, or: python -m pip install 'rerun-sdk==0.22.1'")
            return
        except Exception:
            log.warning("Rerun failed to start; continuing without --view")
            return
        self.rr = rr
        self.enabled = True

    def update(self, app, state, submap, results):
        """Live: trajectory plus the current batch only. The full map waits for relocate()."""
        if not self.enabled:
            return
        chunk = self._snapshot(app, state, submap, results)
        if chunk is not None:
            self._chunks.append(chunk)
            world, cols = self._place(*chunk, results, app)
            if len(world):
                self._log_cloud("world/preview", world, cols)
        self._log_traj(results)

    def relocate(self, app, results):
        """Replace the preview with the full map at the final scale and poses."""
        if not self.enabled:
            return
        self.rr.log("world/preview", self.rr.Clear(recursive=False))
        for chunk in self._chunks:
            world, cols = self._place(*chunk, results, app)
            if len(world):
                self._log_cloud(f"world/map/{chunk[0]:04d}", world, cols)
        self._log_traj(results)

    def hold(self):
        pass  # the Rerun window is its own process

    def _log_cloud(self, path, world, cols):
        self.rr.log(path, self.rr.Points3D(world, colors=(cols * 255).astype(np.uint8)))

    def _log_traj(self, results):
        xyz = np.array([T[:3, 3] for _, T in Graph.poses(results)])
        if len(xyz) > 1:
            self.rr.log("world/traj", self.rr.LineStrips3D([xyz], colors=[SLAM_COLOR]))

    def _snapshot(self, app, state, submap, results):
        pts, confs, cols = submap.get_points(ref_num=0)
        n = submap.H * submap.W
        start = 0 if state.batch_id == 0 else app.batch_overlap
        sl = slice(start * n, submap.S * n)
        keep = confs[sl] > 0
        pts, cols = np.asarray(pts[sl][keep]), np.asarray(cols[sl][keep], dtype=np.float64)
        if not len(pts):
            return None
        if cols.max() > 1:
            cols = cols / 255.0
        _, scale_id = get_base_idx(submap.batch_id, app.batch_size, app.batch_overlap)
        s = float(np.exp(results.atDouble(S(scale_id))))
        pts, cols = _voxel(pts / s, cols, VOXEL)
        return submap.batch_id, pts * s, cols

    def _place(self, batch_id, pts_model, colors, results, app):
        first, scale_id = get_base_idx(batch_id, app.batch_size, app.batch_overlap)
        T = results.atPose3(X(first)).matrix()
        s = float(np.exp(results.atDouble(S(scale_id))))
        world = transform_cloud_by_pose(pts_model / s, T)
        origin = T[:3, 3]
        keep = np.linalg.norm(world - origin, axis=1) <= app.cfg.model.export_max_range
        if self.ceiling is not None:
            zmax = max(T[2, 3] for _, T in Graph.poses(results)) + self.ceiling
            cut = keep & (world[:, 2] <= zmax)
            keep = cut if cut.any() else keep
        return world[keep], colors[keep]


def _voxel(pts, cols, size):
    _, idx = np.unique(np.floor(pts / size).astype(np.int32), axis=0, return_index=True)
    return pts[idx], cols[idx]
