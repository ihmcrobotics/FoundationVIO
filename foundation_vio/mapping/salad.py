"""SALAD place-recognition model (DINOv2 ViT-B/14 + SALAD aggregator), inference only.

Vendored from https://github.com/serizba/salad (GPL-3.0), Izquierdo and Civera, "Optimal Transport Aggregation for
Visual Place Recognition", CVPR 2024. The Sinkhorn solver is adapted there from OpenGlue (MIT). Module and parameter
names match the original so the released dino_salad.ckpt loads with strict=True.
"""

import math
import os

import torch
from torch import nn

DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"  # facebookresearch/dinov2 main, 2026-06-03


def log_otp_solver(log_a, log_b, M, num_iters=20, reg=1.0):  # noqa: N803 - cost matrix
    """Sinkhorn scaling in log space: the log optimal-transport plan for cost M."""
    M = M / reg  # noqa: N806
    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)
    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()
    return M + u.unsqueeze(2) + v.unsqueeze(1)


def get_matching_probs(S, dustbin_score=1.0, num_iters=3, reg=1.0):  # noqa: N803 - score matrix
    """Log assignment probabilities of features to clusters, with a dustbin row."""
    batch_size, m, n = S.size()
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)  # noqa: N806
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score
    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n - m)
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    return log_otp_solver(log_a, log_b, S_aug, num_iters=num_iters, reg=reg) - norm


class SALAD(nn.Module):
    """Sinkhorn Algorithm for Locally Aggregated Descriptors: features + token -> one L2-normalised descriptor."""

    def __init__(self, num_channels=768, num_clusters=64, cluster_dim=128, token_dim=256):
        super().__init__()
        self.num_clusters, self.cluster_dim = num_clusters, cluster_dim
        self.token_features = nn.Sequential(nn.Linear(num_channels, 512), nn.ReLU(), nn.Linear(512, token_dim))
        # nn.Identity stands in for the training-time dropout, keeping the checkpoint's layer indices
        self.cluster_features = nn.Sequential(
            nn.Conv2d(num_channels, 512, 1), nn.Identity(), nn.ReLU(), nn.Conv2d(512, cluster_dim, 1)
        )
        self.score = nn.Sequential(
            nn.Conv2d(num_channels, 512, 1), nn.Identity(), nn.ReLU(), nn.Conv2d(512, num_clusters, 1)
        )
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x, t = x  # feature map [B, C, H/14, W/14], class token [B, C]
        f = self.cluster_features(x).flatten(2)
        p = self.score(x).flatten(2)
        t = self.token_features(t)
        p = torch.exp(get_matching_probs(p, self.dust_bin, 3))[:, :-1, :]
        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)
        f = torch.cat(
            [
                nn.functional.normalize(t, p=2, dim=-1),
                nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1),
            ],
            dim=-1,
        )
        return nn.functional.normalize(f, p=2, dim=-1)


class DINOv2(nn.Module):
    """DINOv2 ViT-B/14 from torch.hub: normalised patch features as a [B, C, H/14, W/14] map, and the class token."""

    num_channels = 768

    def __init__(self):
        super().__init__()
        os.environ["XFORMERS_DISABLED"] = "1"  # If xformers is not available
        # pinned: the hub ref must otherwise be a branch or tag tip, and dinov2 has no tags
        self.model = torch.hub.load(f"facebookresearch/dinov2:{DINOV2_COMMIT}", "dinov2_vitb14", skip_validation=True)

    def forward(self, x):
        B, _, H, W = x.shape  # noqa: N806
        x = self.model.prepare_tokens_with_masks(x)
        for blk in self.model.blocks:
            x = blk(x)
        x = self.model.norm(x)
        f = x[:, 1:].reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)
        return f, x[:, 0]


class VPRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DINOv2()
        self.aggregator = SALAD(num_channels=DINOv2.num_channels)

    def forward(self, x):
        return self.aggregator(self.backbone(x))


def load_model(ckpt_path):
    """The released dino_salad.ckpt on the GPU, in eval mode."""
    model = VPRModel()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.eval().to("cuda")
