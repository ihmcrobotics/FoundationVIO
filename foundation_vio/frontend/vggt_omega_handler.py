"""VGGT-Omega (https://github.com/facebookresearch/vggt-omega, gated on Hugging Face): depth and a pose encoding."""

import numpy as np
import PIL.Image
import torch

from foundation_vio.frontend.base_handler import Prediction, scale_K, unproject, w2c_to_c2w


class VGGTOmegaHandler:
    def __init__(self, image_resolution=512):
        from huggingface_hub import hf_hub_download  # noqa: PLC0415 - heavy imports only when selected
        from vggt_omega.models import VGGTOmega  # noqa: PLC0415

        self.image_resolution = image_resolution
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = VGGTOmega().to(self.device).eval()
        ckpt = hf_hub_download(repo_id="facebook/VGGT-Omega", filename="vggt_omega_1b_512.pt")
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))

    def infer(self, paths, K):
        """VGGT-Omega takes no intrinsics; the calibrated K is used for the unprojection."""
        from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: PLC0415
        from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: PLC0415

        views = load_and_preprocess_images([str(p) for p in paths], image_resolution=self.image_resolution)
        views = views.to(self.device)  # (S, 3, H, W) in [0, 1]
        with torch.inference_mode():
            pred = self.model(views)
        H, W = pred["images"].shape[-2:]
        extrinsics_w2c, _ = encoding_to_camera(pred["pose_enc"], (H, W))
        poses = w2c_to_c2w(extrinsics_w2c[0].cpu().numpy())
        with PIL.Image.open(paths[0]) as first:
            Ks = np.stack([scale_K(K, first.size, (H, W))] * len(paths))
        depth = pred["depth"][0, :, :, :, 0].cpu().numpy()
        pts_cam, points = unproject(depth, Ks, poses)
        rgb = views[0] if views.dim() == 5 else views
        colors = (rgb.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32).clip(0, 1) * 255).round().astype(np.uint8)
        return Prediction(pts_cam, points, poses, pred["depth_conf"][0].cpu().numpy(), depth > 0, colors, Ks)
