"""EthMuSAM: DINOv3-MuSc with cascaded SAM refinement.

Zero-shot method — no training or fine-tuning. All weights are frozen pretrained models.

Architecture:
  Stage 1: DINOv3 patch-feature extraction (layers {6,12,18,24}, 512×512 input)
  Stage 2: LNAMD aggregation (radii r ∈ {1,3,5}) + MSM mutual scoring → coarse heatmap
  Stage 3: Cascaded 3-pass SAM prompting → refined binary segmentation mask
"""

import gc
import re
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from backbone.dinov3_backbone import load_dinov3
from modules._LNAMD import LNAMD
from modules._MSM import MSM

CLASSNAMES = [
    "can", "fabric", "fruit_jelly", "rice",
    "sheet_metal", "vial", "wallplugs", "walnuts",
]


def short_name(backbone_name: str) -> str:
    name = backbone_name.split("/")[-1]
    if "pretrain" in name or "pretrained" in name:
        parts = re.split(r"[-_]", name)
        name = "_".join(p for p in parts if not p.startswith("pretrain"))
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


class EthMuSAM:
    """Zero-shot industrial anomaly detector.

    DINOv3 features scored by MuSc mutual inconsistency mechanism, refined by
    a 3-pass cascaded SAM prompt strategy into precise binary segmentation masks.
    """

    def __init__(
        self,
        backbone_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        img_resize: int = 512,
        device=None,
        features_list: list = None,
        r_list: list = None,
    ):
        self.backbone_name = backbone_name
        self.img_resize = img_resize
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device) if isinstance(device, str) else device
        self.features_list = features_list or [6, 12, 18, 24]
        self.r_list = r_list or [1, 3, 5]
        self.short_name = short_name(backbone_name)
        self.sam_refiner = None

        self.model = load_dinov3(backbone_name, self.device)
        self.preprocess = self.model.get_preprocess(img_resize)

    def load_sam(
        self,
        version: str = "sam3",
        checkpoint_path: str = "models/sam_vit_h.pth",
        model_type: str = "vit_h",
        model_id: str = "models/sam3",
        k_pos: int = 2,
        k_neg: int = 5,
        min_spacing_px: int = 60,
        dilation_kernel: int = 15,
    ):
        """Load SAM refiner. version='sam1' uses segment_anything; 'sam3' uses transformers."""
        from sam_refiner import create_sam_refiner
        if version == "sam1":
            self.sam_refiner = create_sam_refiner(
                "sam1",
                checkpoint_path=checkpoint_path,
                model_type=model_type,
                device=str(self.device),
                k_pos=k_pos, k_neg=k_neg,
                min_spacing_px=min_spacing_px,
                dilation_kernel=dilation_kernel,
            )
        else:
            self.sam_refiner = create_sam_refiner(
                "sam3",
                model_id=model_id,
                device=str(self.device),
                k_pos=k_pos, k_neg=k_neg,
                min_spacing_px=min_spacing_px,
                dilation_kernel=dilation_kernel,
            )

    def _extract_patch_tokens(self, input_img):
        raw = self.model.get_intermediate_layers(
            x=input_img,
            n=[l - 1 for l in self.features_list],
            return_class_token=False,
        )
        fake_cls = [torch.zeros_like(t)[:, :1, :] for t in raw]
        return [torch.cat([fake_cls[i], raw[i]], dim=1) for i in range(len(self.features_list))]

    def infer(self, dataset, batch_size: int = 4, with_masks: bool = False):
        """LNAMD + MSM inference over a dataset.

        Args:
            dataset: torch Dataset returning {'image', 'image_path'[, 'mask']}
            batch_size: DataLoader batch size
            with_masks: if True, also collects GT masks from dataset

        Returns:
            (anomaly_maps, image_paths)           if with_masks=False
            (anomaly_maps, image_paths, gt_masks) if with_masks=True
            anomaly_maps: (N, 1, H, W) float32 numpy
        """
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
        )
        anomaly_maps_r = []
        image_path_list = None
        collected_masks = [] if with_masks else None

        for r_idx, r in enumerate(self.r_list):
            paths_this_r = []
            Z_layers = {}
            LNAMD_r = None

            for batch in tqdm(dataloader, desc=f"r={r}", leave=False):
                image = batch["image"]
                paths_this_r.extend(batch["image_path"])
                if with_masks and r_idx == 0 and "mask" in batch:
                    collected_masks.append(batch["mask"])

                with torch.no_grad(), torch.cuda.amp.autocast():
                    input_img = image.to(torch.float).to(self.device)
                    patch_tokens = self._extract_patch_tokens(input_img)
                    if LNAMD_r is None:
                        LNAMD_r = LNAMD(
                            device=self.device, r=r,
                            feature_dim=patch_tokens[0].shape[-1],
                            feature_layer=self.features_list,
                        )
                    features = LNAMD_r._embed(patch_tokens)
                    features /= features.norm(dim=-1, keepdim=True)

                for l in range(len(self.features_list)):
                    key = str(l)
                    if key not in Z_layers:
                        Z_layers[key] = []
                    Z_layers[key].append(features[:, :, l, :])

                del patch_tokens, features
                torch.cuda.empty_cache()

            if image_path_list is None:
                image_path_list = paths_this_r
            del LNAMD_r
            gc.collect()

            maps_per_layer = []
            for l_key in sorted(Z_layers.keys()):
                Z = torch.cat(Z_layers[l_key], dim=0).to(self.device)
                del Z_layers[l_key]
                torch.cuda.empty_cache()
                maps_msm = MSM(Z=Z, device=self.device, topmin_min=0, topmin_max=0.3)
                maps_per_layer.append(maps_msm.cpu().float())
                del Z, maps_msm
                torch.cuda.empty_cache()

            del Z_layers
            gc.collect()
            anomaly_maps_r.append(torch.stack(maps_per_layer, dim=0).mean(0))
            del maps_per_layer
            gc.collect()

        anomaly_maps = torch.stack(anomaly_maps_r, dim=0).mean(0).to(self.device)
        del anomaly_maps_r
        B, L = anomaly_maps.shape
        H = int(np.sqrt(L))
        anomaly_maps = F.interpolate(
            anomaly_maps.view(B, 1, H, H), size=self.img_resize,
            mode="bilinear", align_corners=True,
        )
        result = anomaly_maps.cpu().float().numpy()
        del anomaly_maps
        torch.cuda.empty_cache()
        gc.collect()

        if with_masks:
            gt = torch.cat(collected_masks, dim=0).numpy() if collected_masks else None
            return result, image_path_list, gt
        return result, image_path_list

    def refine_with_sam(self, anomaly_maps, image_paths):
        """Gate anomaly heatmaps with SAM binary masks (no-op if SAM not loaded).

        Falls back to raw heatmap when SAM returns an empty mask.
        """
        if self.sam_refiner is None:
            return anomaly_maps
        refined = []
        for amap, path in tqdm(zip(anomaly_maps, image_paths),
                               total=len(anomaly_maps), desc="SAM refine"):
            img_np = np.array(
                Image.open(path).convert("RGB").resize(
                    (self.img_resize, self.img_resize), Image.BILINEAR
                )
            )
            hmap = amap.squeeze()
            m3 = self.sam_refiner.refine(img_np, hmap)
            refined_map = hmap * m3.astype(np.float32) if m3.sum() > 0 else hmap
            refined.append(refined_map.astype(np.float32)[np.newaxis])
        return np.stack(refined, axis=0)

    def save_submission(
        self,
        anomaly_maps,
        image_paths,
        category: str,
        split: str,
        submission_dir,
        threshold: float,
    ):
        """Write float16 tiff + thresholded binary png for one category/split.

        Layout matches MVTec AD 2 benchmark submission format:
          anomaly_images/{category}/{split}/{stem}.tiff
          anomaly_images_thresholded/{category}/{split}/{stem}.png
        """
        submission_dir = Path(submission_dir)
        tiff_dir = submission_dir / "anomaly_images" / category / split
        png_dir  = submission_dir / "anomaly_images_thresholded" / category / split
        tiff_dir.mkdir(parents=True, exist_ok=True)
        png_dir.mkdir(parents=True, exist_ok=True)

        for amap, img_path in zip(anomaly_maps, image_paths):
            stem = Path(img_path).stem
            amap_2d = amap.squeeze()
            tifffile.imwrite(str(tiff_dir / f"{stem}.tiff"), amap_2d.astype(np.float16))
            binary = (amap_2d >= threshold).astype(np.uint8) * 255
            Image.fromarray(binary, mode="L").save(str(png_dir / f"{stem}.png"))

        print(f"    saved {len(anomaly_maps)} maps → {tiff_dir}")
