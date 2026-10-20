"""Depth Anything 3 (https://github.com/bytedance-seed/depth-anything-3): depth, confidence and poses per image."""

import numpy as np
import PIL.Image
import torch

from foundation_vio.frontend.base_handler import Prediction, scale_K, unproject, w2c_to_c2w


class DepthAnything3Handler:
    def __init__(self, model_id, process_res=504):
        from depth_anything_3.api import DepthAnything3  # noqa: PLC0415 - heavy import only when selected

        self.process_res = process_res
        self.model = DepthAnything3.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu").eval()

    def infer(self, paths, K):
        """Conditioned on the calibrated K (DA3 ignores it without extrinsics) and unprojected with it."""
        images = [PIL.Image.open(str(p)).convert("RGB") for p in paths]
        pred = self.model.inference(
            image=images,
            intrinsics=np.stack([K] * len(images)).astype(np.float32),
            process_res=self.process_res,
            ref_view_strategy="first",  # DA3's default is "saddle_balanced"
        )
        Ks = np.stack([scale_K(K, images[0].size, pred.depth.shape[-2:])] * len(images))
        poses = w2c_to_c2w(pred.extrinsics)  # DA3 returns (S, 3, 4) despite documenting (S, 4, 4)
        depth = pred.depth.astype(np.float32)
        pts_cam, points = unproject(depth, Ks, poses)
        colors = np.asarray(pred.processed_images).astype(np.uint8)
        return Prediction(pts_cam, points, poses, pred.conf.astype(np.float32), depth > 0, colors, Ks)
