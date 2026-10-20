"""FoundationVIO: model relative poses and IMU preintegration in an incremental iSAM2 graph (driven by run.py)."""

import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import gtsam
import numpy as np
import torch
import yaml
from gtsam import Pose3
from gtsam.symbol_shorthand import B, S, V, X
from tqdm import tqdm

from foundation_vio.frontend.depth_anything3_handler import DepthAnything3Handler
from foundation_vio.frontend.filter import apply_keep_mask, baseline_filter_mask, edge_filter_mask
from foundation_vio.frontend.vggt_omega_handler import VGGTOmegaHandler
from foundation_vio.graph.graph import HUBER_K, Graph
from foundation_vio.io.load import make_loader
from foundation_vio.io.results import export_submaps, write_tum
from foundation_vio.mapping.fusion import compute_fusion_roots
from foundation_vio.mapping.loop_closure import ImageRetrieval
from foundation_vio.mapping.map import GlobalMap
from foundation_vio.mapping.submap import Submap
from foundation_vio.math import irls, vi_init
from foundation_vio.math.matrix import get_base_idx, se3_inverse, transform_cloud_by_pose

log = logging.getLogger(__name__)


def _namespace(d):
    """Nested dict -> attribute access (cfg.model.batch_size)."""
    return SimpleNamespace(**{k: _namespace(v) if isinstance(v, dict) else v for k, v in d.items()})


@dataclass
class IterState:
    isam: gtsam.ISAM2
    pbar: tqdm
    T_world_prev: np.ndarray  # pose of the next batch's first image
    v_world_prev: np.ndarray  # its velocity
    batch_id: int = 0
    prev_submap: Submap | None = None
    accum_scale_ratio: float = 1.0


class FoundationVIO:
    def __init__(self, config):
        self.config = config
        self.cfg = _namespace(config)
        self.batch_size, self.batch_overlap = self.cfg.model.batch_size, self.cfg.model.batch_overlap
        self.stride = self.batch_size - self.batch_overlap
        self.b0 = gtsam.imuBias.ConstantBias()  # zero initial bias

        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # noqa: DTZ005 - local wall-clock folder name
        self.result_path = Path("results") / self.cfg.sequence.sequence_name / run_name
        self.result_path.mkdir(parents=True, exist_ok=True)
        print(f"RESULT PATH: {self.result_path}")
        self._setup_logging(self.cfg.logging.logging_level)
        with (self.result_path / "config.yaml").open("w") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

        self.graph = Graph()
        self.global_map = GlobalMap()
        self.initial_estimate = gtsam.Values()
        # one trajectory per run: SLAM with loop closures, odometry without
        self.loop_detector = ImageRetrieval() if self.cfg.loop_closure.enabled else None
        self._imu_time_ns = None
        self.viewer = None  # set by run.py --view
        irls.warmup()

    def _setup_logging(self, level):
        """foundation_vio.* logs go to foundation_vio.log in the run directory; warnings and errors also to stderr."""
        logger = logging.getLogger("foundation_vio")
        logger.setLevel(level)
        file_handler = logging.FileHandler(self.result_path / "foundation_vio.log")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(logging.WARNING)
        logger.handlers = [file_handler, console]

    def load(self):
        name = self.cfg.model.use_model
        if name == "DepthAnything3":
            self.handler = DepthAnything3Handler(self.cfg.model.model_id or "depth-anything/DA3NESTED-GIANT-LARGE-1.1")
        elif name == "VGGTOmega":
            self.handler = VGGTOmegaHandler()
        else:
            raise ValueError(f"model.use_model must be 'DepthAnything3' or 'VGGTOmega', got {name!r}")
        self.loader = make_loader(self.config)
        self.loader.fetch_image_from_dir(img_stride=self.cfg.sequence.img_stride)
        self.t_ns = [self.loader.get_timestamp_ns(p) for p in self.loader.image_path_list]
        self.graph.set_imu_params(*self.loader.fetch_imu_params())
        self.imu_df = self.loader.fetch_imu_data()
        self.loader.fetch_T_cam_imu()
        self.T_BC, self.T_CB = self.loader.T_BC, self.loader.T_CB
        self.T_BC_f32 = self.T_BC.astype(np.float32)  # a float64 transform would upcast every point cloud
        self.timeshift_cam_imu = self.loader.fetch_timeshift_cam_imu()
        self.intrinsics = self.loader.fetch_T_cam_intrinsics()

    # --- initialisation ---

    def _set_initial(self):
        """Priors on the first nodes; X(0)'s rotation from the VI-init gravity. Returns the initial pose."""
        imu = self.cfg.imu
        T_initial = np.eye(4)
        if imu.use_imu:
            result = vi_init.run(
                self.loader,
                self.imu_df,
                self.timeshift_cam_imu,
                self._vi_init_window,
                window_frames=self.batch_size + self.stride,
            )
            if result is not None:
                T_initial[:3, :3] = result.dR

        # roll/pitch get a soft prior the IMU keeps refining; yaw and translation are pinned (rx ry rz tx ty tz)
        rp = imu.vi_init.first_pose_rp_sigma
        self.graph.add_pose_prior(0, T_initial, np.array([rp, rp] + [1e-5] * 4))
        if imu.use_imu:
            self.graph.add_vel_prior(0, np.zeros(3), np.array([1e-1] * 3))
            self.graph.add_bias_prior(0, self.b0, np.array(imu.bias_prior_sigmas))
        self.initial_estimate.insert(S(0), 0.0)  # log 1, loose enough for the scene's model-unit scale to settle
        self.graph.add_scale_prior(0, 0.0, imu.vi_init.scale_sigma)
        return T_initial

    def prep_iterate(self):
        """Initialise, create iSAM2 and return (state, total_batches)."""
        T_initial = self._set_initial()
        n_batches = int(np.ceil(len(self.loader.image_path_list) / self.stride))
        cap = getattr(self.cfg.sequence, "max_iter", None)
        total_batches = n_batches if cap is None else min(cap, n_batches)
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(0.01)
        params.relinearizeSkip = 1
        params.enablePartialRelinearizationCheck = True
        pbar = tqdm(total=total_batches, desc="Processing batches", unit="batch")
        return IterState(gtsam.ISAM2(params), pbar, T_initial, np.zeros(3)), total_batches

    # --- per batch ---

    def iterate(self, state: IterState) -> IterState | None:
        """Process one batch. Returns the updated state, or None once too few images remain."""
        base, _ = get_base_idx(state.batch_id, self.batch_size, self.batch_overlap)
        paths = self.loader.get_batch_from_images(self.loader.image_path_list, base, self.batch_size)
        if len(paths) < self.batch_size:
            return None

        submap = self._run_model(paths, state.batch_id)
        self.global_map.add_submap(submap)
        rel_scale = self._estimate_scale(state, submap)
        state.accum_scale_ratio *= rel_scale
        scale_id, pims = self._add_factors(state, submap, rel_scale)
        self._add_initial_estimates(state, submap, scale_id, pims)
        results = self._update_isam(state)
        if self.loop_detector is not None and self._check_loop_closure(submap, results, state.isam):
            self._converge_isam(state.isam, max_iters=5)
            results = state.isam.calculateEstimate()

        next_id = base + self.stride
        T_next = results.atPose3(X(next_id)).matrix()
        v_next = results.atVector(V(next_id)) if self.cfg.imu.use_imu else None
        if self.viewer is not None:
            self.viewer.update(self, state, submap, results)

        self.initial_estimate.clear()
        self.graph.graph.resize(0)
        state.batch_id += 1
        state.pbar.update()
        state.prev_submap, state.T_world_prev, state.v_world_prev = submap, T_next, v_next
        return state

    def _new_image_offsets(self, state):
        """Offsets within the batch of the images that get new nodes (the overlap images already have them)."""
        return range(1 if state.batch_id == 0 else self.batch_overlap, self.batch_size)

    def _to_body(self, pred):
        """Model points and camera poses in the IMU (body) frame."""
        return transform_cloud_by_pose(pred.points, self.T_BC_f32), [self.T_BC @ T @ self.T_CB for T in pred.poses]

    def _vi_init_window(self, first, last):
        """Chained model body poses for images first..last (unfiltered points), for VI-init."""

        def infer(paths):
            pred = self.handler.infer(paths, self.intrinsics)
            return (*self._to_body(pred), pred.confs)

        return vi_init.model_window(
            self.loader, infer, first, last, batch_size=self.batch_size, batch_overlap=self.batch_overlap
        )

    def _model_submap(self, submap_id, paths, *, lc=False):
        """Run the model on paths: a submap of body poses and filtered body-frame points, and the raw prediction."""
        pred = self.handler.infer(paths, self.intrinsics)
        points, T_local_bodies = self._to_body(pred)
        points, confs = self._filter_points(pred, points, label=f"submap {submap_id}")
        t_ns = [self.loader.get_timestamp_ns(p) for p in paths]
        return Submap(submap_id, t_ns, T_local_bodies, points, confs, pred.colors, lc=lc), pred, confs

    def _run_model(self, paths, batch_id):
        submap, pred, confs = self._model_submap(batch_id, paths)
        try:  # only the grouping is stored and applied at export: the estimator sees the same points either way
            submap.fusion_roots = compute_fusion_roots(pred.pts_cam, confs, np.asarray(pred.poses), pred.K)
        except Exception:
            log.exception(f"[fusion] submap {batch_id}: matching failed, exporting unfused")
        if self.loop_detector is not None:
            submap.retrieval_vectors = self.loop_detector.submap_descriptors(submap)
        return submap

    def _filter_points(self, pred, points, *, label):
        """Zero depth-edge pixels and, with model.filter_distance, points beyond 5x the batch baseline."""
        valid = pred.masks
        keep = valid & edge_filter_mask(pred.pts_cam, valid_mask=valid)
        if self.cfg.model.filter_distance:
            keep &= baseline_filter_mask(pred.pts_cam, pred.poses)
        for s in range(keep.shape[0]):  # a degenerate model baseline can wipe whole frames: keep those unfiltered
            if keep[s].sum() < 0.01 * max(valid[s].sum(), 1):
                keep[s] = valid[s]
                log.warning(f"[filter] {label}: frame {s} nearly empty after filtering, reverted to unfiltered")
        return apply_keep_mask(points, pred.confs, keep)

    def _estimate_scale(self, state, submap):
        """Scale of this submap relative to the previous one, from IRLS on their shared overlap image."""
        if state.prev_submap is None:
            return 1.0
        prev = state.prev_submap.get_points(frame_num=-self.batch_overlap, ref_num=-self.batch_overlap)
        curr = submap.get_points(frame_num=0, ref_num=0)
        ratio = irls.estimate_scale(prev[0], prev[1], curr[0], curr[1])
        # only loop closure and export need the previous submap's points again, and they reload them
        state.prev_submap.spill(self.result_path / "point_cache" / f"{state.prev_submap.batch_id:06d}")
        return 1 / ratio

    def _imu_segment(self, prev_img_time, curr_img_time):
        """Copy of the imu_df rows with prev <= time - timeshift <= curr."""
        if self._imu_time_ns is None or len(self._imu_time_ns) != len(self.imu_df):
            self._imu_time_ns = self.imu_df["time"].to_numpy()
            assert np.all(np.diff(self._imu_time_ns) >= 0), "imu_df.time must be sorted"
        lo = np.searchsorted(self._imu_time_ns, prev_img_time + self.timeshift_cam_imu, side="left")
        hi = np.searchsorted(self._imu_time_ns, curr_img_time + self.timeshift_cam_imu, side="right")
        return self.imu_df.iloc[lo:hi].copy()

    def _add_factors(self, state, submap, rel_scale):
        """IMU and scale-pose factors per new image, a scale-scale factor to the previous batch; (scale id, pims)."""
        std = self.cfg.std
        base, scale_id = get_base_idx(state.batch_id, self.batch_size, self.batch_overlap)
        pims = {}
        for off in self._new_image_offsets(state):
            i = base + off
            if self.cfg.imu.use_imu:
                seg = self._imu_segment(submap.t_ns[off - 1], submap.t_ns[off])
                if len(seg) == 0:  # an empty segment makes a rank-deficient IMU factor
                    log.warning(f"IMU segment empty for X({i}), using a constant-velocity factor")
                    self.graph.add_vel_factor(i - 1, i, np.ones(3))
                else:
                    pims[i] = self.graph.preintegrate(seg, self.b0)
                    self.graph.add_imu_factor(pims[i], i - 1, i)
            T_rel = se3_inverse(submap.T_local_bodies[off - 1]) @ submap.T_local_bodies[off]
            self.graph.add_scale_pose_factor(
                i - 1, i, scale_id, T_rel, sigma_trans=std.custom_trans, sigma_rot=np.deg2rad(std.custom_rot)
            )
        prev_scale_id = 0 if state.batch_id == 0 else scale_id - self.stride
        sigma = 1e-4 if state.batch_id == 0 else std.scale
        self.graph.add_scale_factor(prev_scale_id, scale_id, float(np.log(rel_scale)), sigma)
        return scale_id, pims

    def _add_initial_estimates(self, state, submap, scale_id, pims):
        """Seed the new nodes: scale from the accumulated IRLS ratios, poses by IMU propagation (or model chaining)."""
        use_imu = self.cfg.imu.use_imu
        base, _ = get_base_idx(state.batch_id, self.batch_size, self.batch_overlap)
        self.initial_estimate.insert(S(scale_id), np.log(state.accum_scale_ratio))
        if state.batch_id == 0:
            self.initial_estimate.insert(X(0), Pose3(state.T_world_prev))
            if use_imu:
                self.initial_estimate.insert(V(0), state.v_world_prev)
                self.initial_estimate.insert(B(0), self.b0)

        nav = gtsam.NavState(Pose3(state.T_world_prev), state.v_world_prev) if use_imu else None
        T = state.T_world_prev
        for off in self._new_image_offsets(state):
            i = base + off
            if use_imu:
                if i in pims:  # no preintegration (empty IMU segment): hold the previous state
                    nav = pims[i].predict(nav, self.b0)
                self.initial_estimate.insert(X(i), nav.pose())
                self.initial_estimate.insert(V(i), nav.velocity())
                self.initial_estimate.insert(B(i), self.b0)
            else:
                T = T @ se3_inverse(submap.T_local_bodies[off - 1]) @ submap.T_local_bodies[off]
                self.initial_estimate.insert(X(i), Pose3(T))

    def _update_isam(self, state):
        """Push this batch's VIO factors; on failure keep the last committed estimate."""
        try:
            state.isam.update(self.graph.graph, self.initial_estimate)
        except Exception:
            log.exception("iSAM2 update failed")
        return state.isam.calculateEstimate()

    # --- loop closure ---

    def _image_index(self, submap_id, frame):
        return get_base_idx(submap_id, self.batch_size, self.batch_overlap)[0] + frame

    def _check_loop_closure(self, submap, results, isam):
        """Retrieve revisits of this submap; for each, run the model on the image pair and apply a closure."""
        lc = self.cfg.loop_closure
        matches = self.loop_detector.find_loop_closures(
            self.global_map,
            submap,
            max_similarity_thres=lc.similarity_thres,
            max_loop_closures=lc.max_per_batch,
            max_matches_per_pair=1,
            min_submap_gap=lc.min_submap_gap,
        )
        applied = 0
        for match in matches:
            query_id = self._image_index(match.query_submap_id, match.query_submap_frame)
            match_id = self._image_index(match.detected_submap_id, match.detected_submap_frame)
            lc_submap_id = match.query_submap_id * 10000 + match.detected_submap_id  # unique below 10000 submaps
            paths = [self.loader.image_path_list[query_id], self.loader.image_path_list[match_id]]
            lc_submap, _, _ = self._model_submap(lc_submap_id, paths, lc=True)
            self.global_map.add_submap(lc_submap)
            if self._add_loop_closure(lc_submap, match, query_id, match_id, results, isam):
                applied += 1
        if applied:
            log.info(f"batch {submap.batch_id}: applied {applied} loop-closure bundle(s)")
        return applied

    def _add_loop_closure(self, lc_submap, match, query_id, match_id, results, isam):
        """Link X(query), X(match) via a new scale node tied to both batches by Huber IRLS ratios; applied now."""
        std = self.cfg.std
        query = self.global_map.submaps[match.query_submap_id]
        matched = self.global_map.submaps[match.detected_submap_id]
        spilled = [s for s in {query, matched} if s.points is None]
        for s in spilled:
            s.reload()
        q_ratio = self._lc_scale_ratio(query, match.query_submap_frame, lc_submap, 0)
        m_ratio = self._lc_scale_ratio(matched, match.detected_submap_frame, lc_submap, 1)
        for s in spilled:
            s.release_points()

        _, q_scale_id = get_base_idx(match.query_submap_id, self.batch_size, self.batch_overlap)
        _, m_scale_id = get_base_idx(match.detected_submap_id, self.batch_size, self.batch_overlap)
        lc_scale_id = lc_submap.batch_id
        lc_graph, lc_values = gtsam.NonlinearFactorGraph(), gtsam.Values()
        lc_values.insert(S(lc_scale_id), results.atDouble(S(q_scale_id)) + float(np.log(q_ratio)))
        noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(HUBER_K),
            gtsam.noiseModel.Diagonal.Sigmas(np.array([std.lc_scale])),
        )
        lc_graph.add(gtsam.BetweenFactorDouble(S(q_scale_id), S(lc_scale_id), float(np.log(q_ratio)), noise))
        lc_graph.add(gtsam.BetweenFactorDouble(S(m_scale_id), S(lc_scale_id), float(np.log(m_ratio)), noise))
        T_lc = se3_inverse(lc_submap.T_local_bodies[0]) @ lc_submap.T_local_bodies[1]
        self.graph.add_scale_pose_factor(
            query_id,
            match_id,
            lc_scale_id,
            T_lc,
            sigma_rot=np.deg2rad(std.lc_rot),
            sigma_trans=std.lc_trans,
            target_graph=lc_graph,
        )
        try:
            isam.update(lc_graph, lc_values)
        except Exception:
            log.exception("loop-closure bundle failed; skipping it")
            return False
        return True

    @staticmethod
    def _lc_scale_ratio(submap, frame, lc_submap, lc_frame):
        pts, confs, _ = submap.get_points(frame_num=frame, ref_num=frame)
        lc_pts, lc_confs, _ = lc_submap.get_points(frame_num=lc_frame, ref_num=lc_frame)
        return irls.estimate_scale(pts, confs, lc_pts, lc_confs)

    # --- final solve and export ---

    @staticmethod
    def _converge_isam(isam, max_iters=10):
        """Plain isam.update() calls until nothing relinearises (at most max_iters); on failure keep the last estimate."""
        try:
            for _ in range(max_iters):
                if isam.update().getVariablesRelinearized() == 0:
                    break
        except Exception:
            log.exception("isam.update() convergence loop failed; keeping the last good estimate")

    def optimize(self, state):
        """Converge the live iSAM (it already holds any loop closures)."""
        self._converge_isam(state.isam)
        return state.isam.calculateEstimate()

    def save_data(self, final_result):
        """Write the trajectory (SLAM or odometry, per loop_closure.enabled), the batch scales and the per-image clouds."""
        index, poses = zip(*Graph.poses(final_result), strict=True)
        write_tum(self.result_path / "tum_poses.csv", [self.t_ns[i] for i in index], poses)
        self.global_map.set_optimised(final_result, self.batch_size, self.batch_overlap)
        self.global_map.scales().to_csv(self.result_path / "submap_scales.csv", index=False)

        torch.cuda.empty_cache()  # the model is done: free VRAM before the export
        export_submaps(
            self.global_map,
            self.result_path / "slam_clouds_downsampled",
            self.t_ns,
            self.stride,
            max_range=self.cfg.model.export_max_range,
            conf_percentile=self.cfg.model.export_conf_percentile,
        )
        shutil.rmtree(self.result_path / "point_cache", ignore_errors=True)
        print(f"Trajectories and clouds saved to {self.result_path}")
