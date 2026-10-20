"""All submaps, their place-recognition search and their optimised poses and scales."""

import numpy as np
import open3d as o3d
import pandas as pd
import torch
from gtsam.symbol_shorthand import S, X

from foundation_vio.math.matrix import get_base_idx


class GlobalMap:
    def __init__(self):
        self.submaps = {}
        self._retrieval_cache = None  # (cache key, E (M, D), row_submap (M,), row_frame (M,))

    def add_submap(self, submap):
        self.submaps[submap.batch_id] = submap

    def batches(self):
        return [self.submaps[k] for k in sorted(self.submaps) if not self.submaps[k].lc]

    def set_optimised(self, values, batch_size, batch_overlap):
        """World pose of every batch's frame 0 and its metric scale (exp of its S node) from the final estimate."""
        for submap in self.batches():
            first, scale_id = get_base_idx(submap.batch_id, batch_size, batch_overlap)
            submap.T_world_body0 = values.atPose3(X(first)).matrix()
            submap.apply_scale(np.exp(values.atDouble(S(scale_id))))

    def scales(self):
        """Per batch: first image time, model-units-per-metre scale, model and metric first-last translation."""
        rows = []
        for m in self.batches():
            trans = np.linalg.norm(m.T_local_bodies[-1][:3, 3])  # already metric
            rows.append(
                [m.batch_id, *divmod(m.t_ns[0], 1_000_000_000), float(m.scale), float(trans), float(trans * m.scale)]
            )
        return pd.DataFrame(rows, columns=["#batch_id", "sec", "nsec", "opt_b2m_scale", "opt_rel_trans", "rel_trans"])

    def retrieve_best_score_frame(self, query_vector, current_submap_id, *, min_submap_gap):
        """(distance, submap, frame) of the nearest descriptor outside the last min_submap_gap submaps; 1000 if none."""
        E, row_submap, row_frame = self._retrieval_matrix()
        if E is None:
            return 1000, 0, 0
        batch_ids = [m.batch_id for m in self.batches()]  # the current submap is the last one
        skip = set(batch_ids[-min_submap_gap:]) | {current_submap_id}
        scores = torch.linalg.norm(E - query_vector.reshape(1, -1), dim=1).cpu().numpy()
        scores[np.isin(row_submap, list(skip))] = np.inf
        best = int(np.argmin(scores))  # rows ordered by (submap, frame): first minimum wins ties
        if scores[best] >= 1000:
            return 1000, 0, 0
        return float(scores[best]), int(row_submap[best]), int(row_frame[best])

    def _retrieval_matrix(self):
        """Cached stack of all batch frame descriptors; Nones when empty."""
        subs = self.batches()
        cache_key = tuple((m.batch_id, id(m.retrieval_vectors)) for m in subs)
        if self._retrieval_cache is None or self._retrieval_cache[0] != cache_key:
            if not subs:
                self._retrieval_cache = (cache_key, None, None, None)
            else:
                chunks = [m.retrieval_vectors for m in subs]
                E = torch.cat([torch.as_tensor(c).reshape(len(c), -1) for c in chunks], dim=0)
                row_submap = np.repeat([m.batch_id for m in subs], [len(c) for c in chunks])
                row_frame = np.concatenate([np.arange(len(c)) for c in chunks])
                self._retrieval_cache = (cache_key, E, row_submap, row_frame)
        return self._retrieval_cache[1:]


def write_lc_candidates_ply(path, positions, lc_candidates):
    """Trajectory (black) and loop-closure candidate edges (red) as a LineSet PLY; positions: frame -> (x, y, z)."""
    frames = sorted(positions)
    points = [positions[f] for f in frames]
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    colors = [[0.0, 0.0, 0.0]] * len(lines)
    for query, match, _ in lc_candidates:
        if query in positions and match in positions:
            points += [positions[query], positions[match]]
            lines.append([len(points) - 2, len(points) - 1])
            colors.append([1.0, 0.0, 0.0])
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64).reshape(-1, 3)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32).reshape(-1, 2)),
    )
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64).reshape(-1, 3))
    o3d.io.write_line_set(str(path), line_set)
