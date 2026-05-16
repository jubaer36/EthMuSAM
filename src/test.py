"""Evaluate EthMuSAM on MVTec AD 2 test_public split (SegF1 metric).

Threshold must be calibrated on an independent dataset to avoid leakage from
the competition test splits. Use scripts/find_threshold_mvtec1.py or
scripts/find_threshold_visa.py to find the threshold, then pass it here.

Usage:
    python src/test.py --threshold 0.357525 --data_path ./data/mvtec_ad_2/
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.model import EthMuSAM, CLASSNAMES
import datasets.mvtec_ad2 as mvtec_ad2
from utils.metrics import compute_segf1_at_threshold

import warnings
warnings.filterwarnings("ignore")


def _build_sam(model, args):
    if not args.use_sam:
        return
    if args.sam_version == "sam1":
        model.load_sam(
            "sam1",
            checkpoint_path=args.sam_checkpoint,
            model_type=args.sam_model_type,
            k_pos=args.sam_k_pos, k_neg=args.sam_k_neg,
            min_spacing_px=args.sam_spacing, dilation_kernel=args.sam_dilation,
        )
    else:
        model.load_sam(
            "sam3",
            model_id=args.sam3_model_id,
            k_pos=args.sam_k_pos, k_neg=args.sam_k_neg,
            min_spacing_px=args.sam_spacing, dilation_kernel=args.sam_dilation,
        )
    print(f"SAM{args.sam_version[-1]} refiner loaded.")


def main():
    parser = argparse.ArgumentParser(description="EthMuSAM — evaluate on MVTec AD 2 test_public")
    parser.add_argument("--data_path",      default="./data/mvtec_ad_2/")
    parser.add_argument("--classes",        nargs="+", default=CLASSNAMES)
    parser.add_argument("--backbone_name",  default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--img_resize",     type=int, default=512)
    parser.add_argument("--feature_layers", type=int, nargs="+", default=[5, 11, 17, 23])
    parser.add_argument("--r_list",         type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--batch_size",     type=int, default=4)
    parser.add_argument("--device",         type=int, default=0)
    parser.add_argument("--threshold",      type=float, required=True,
                        help="Fixed threshold (calibrate via scripts/find_threshold_*.py).")
    parser.add_argument("--use_sam",        action="store_true", default=True)
    parser.add_argument("--sam_version",    default="sam3", choices=["sam1", "sam3"])
    parser.add_argument("--sam_checkpoint", default="models/sam_vit_h.pth")
    parser.add_argument("--sam_model_type", default="vit_h")
    parser.add_argument("--sam3_model_id",  default="models/sam3")
    parser.add_argument("--sam_k_pos",      type=int, default=2)
    parser.add_argument("--sam_k_neg",      type=int, default=5)
    parser.add_argument("--sam_spacing",    type=int, default=60)
    parser.add_argument("--sam_dilation",   type=int, default=15)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    print(f"Backbone  : {args.backbone_name}")
    print(f"Device    : {device}")
    print(f"Threshold : {args.threshold}")

    model = EthMuSAM(
        backbone_name=args.backbone_name,
        img_resize=args.img_resize,
        device=device,
        features_list=[l + 1 for l in args.feature_layers],
        r_list=args.r_list,
    )
    _build_sam(model, args)

    cat_maps = {}
    cat_gt   = {}

    print("\n" + "=" * 60)
    print("Inference on MVTec AD 2 test_public")
    print("=" * 60)

    for category in args.classes:
        print(f"\n[{category}]")
        try:
            dataset = mvtec_ad2.MVTecAD2Dataset(
                source=args.data_path,
                classname=category,
                split=mvtec_ad2.DatasetSplit.TEST,
                resize=args.img_resize,
                imagesize=args.img_resize,
                clip_transformer=model.preprocess,
            )
        except Exception as e:
            print(f"  Skipping: {e}")
            continue
        if len(dataset) == 0:
            print("  No images.")
            continue

        anomaly_maps, image_paths, gt_masks = model.infer(
            dataset, batch_size=args.batch_size, with_masks=True
        )
        if gt_masks is None or gt_masks.sum() == 0:
            print("  No GT mask pixels — skipping.")
            continue

        anomaly_maps = model.refine_with_sam(anomaly_maps, image_paths)
        cat_maps[category] = anomaly_maps
        cat_gt[category]   = gt_masks.astype(np.int32)

        del dataset, gt_masks
        gc.collect()
        torch.cuda.empty_cache()

    if not cat_maps:
        print("ERROR: no valid categories found.")
        return

    print(f"\n{'='*52}")
    print(f"{'Category':14s}  {'Threshold':>9s}  {'SegF1':>8s}")
    print(f"{'='*52}")
    segf1_ls = []
    for cat in args.classes:
        if cat not in cat_maps:
            print(f"  {cat} (skipped)")
            continue
        segf1 = compute_segf1_at_threshold(cat_gt[cat], cat_maps[cat], args.threshold)
        segf1_ls.append(segf1)
        print(f"{cat:14s}  {args.threshold:>9.4f}  {segf1*100:>7.2f}%")

    if segf1_ls:
        print(f"{'-'*52}")
        print(f"{'mean':14s}  {args.threshold:>9.4f}  {sum(segf1_ls)/len(segf1_ls)*100:>7.2f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
