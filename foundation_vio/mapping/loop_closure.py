"""Place recognition for loop closure: SALAD descriptors (DINOv2 backbone) per frame, nearest past frame by L2."""

import warnings
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import torch
import torchvision.transforms as T  # noqa: N812 - standard torchvision alias

from foundation_vio.mapping.salad import load_model

SALAD_CKPT_URL = "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"


class LoopMatch(NamedTuple):
    similarity_score: float  # descriptor distance, lower is more similar
    query_submap_id: int
    query_submap_frame: int
    detected_submap_id: int
    detected_submap_frame: int


class ImageRetrieval:
    def __init__(self, input_size=224):
        ckpt = Path(torch.hub.get_dir()) / "checkpoints/dino_salad.ckpt"
        if not ckpt.exists():  # first run: fetch the released SALAD weights (350 MB)
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(SALAD_CKPT_URL, str(ckpt))
        with warnings.catch_warnings():  # torch.hub dinov2 warns "xFormers is (not) available" on import
            warnings.filterwarnings("ignore", message="xFormers is", category=UserWarning)
            self.model = load_model(str(ckpt)).eval()
        self.to_pil = T.ToPILImage()
        self.transform = T.Compose(
            [
                T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def submap_descriptors(self, submap):
        """One descriptor per frame of the submap's images."""
        with torch.no_grad():
            imgs = torch.stack([self.transform(self.to_pil(img)) for img in submap.images()])
            return self.model(imgs.to("cuda"))

    @staticmethod
    def find_loop_closures(
        global_map, submap, *, max_similarity_thres, max_loop_closures, max_matches_per_pair, min_submap_gap
    ):
        """Per query frame, the closest frame of an older submap within the threshold; the best max_loop_closures."""
        matches = []
        pair_counts = defaultdict(int)
        for frame, vector in enumerate(submap.retrieval_vectors):
            score, match_id, match_frame = global_map.retrieve_best_score_frame(
                vector, submap.batch_id, min_submap_gap=min_submap_gap
            )
            if score < max_similarity_thres and pair_counts[match_id] < max_matches_per_pair:
                matches.append(LoopMatch(score, submap.batch_id, frame, match_id, match_frame))
                pair_counts[match_id] += 1
        return sorted(matches)[:max_loop_closures]
