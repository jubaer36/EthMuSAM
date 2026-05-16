#!/usr/bin/env python3
"""Evaluate SegF1 at a fixed threshold across MVTec AD1, VisA, and MVTec AD2.

Threshold should be calibrated on an independent dataset (MVTec AD1 or VisA)
using scripts/find_threshold_mvtec1.py or scripts/find_threshold_visa.py.

Usage:
    python scripts/eval_segf1_all.py --threshold 0.357525
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.model import EthMuSAM
from utils.metrics import compute_metrics, compute_segf1_at_threshold

import warnings
warnings.filterwarnings("ignore")

_MVTEC1_CLASSES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]
_VISA_CLASSES = [
    "candle", "capsules", "cashew", "chewinggum",
    "fryum", "macaroni1", "macaroni2", "pcb1",
    "pcb2", "pcb3", "pcb4", "pipe_fryum",
]
_MVTEC2_CLASSES = [
    "can", "fabric", "fruit_jelly", "rice",
    "sheet_metal", "vial", "wallplugs", "walnuts",
]


def _build_sam(model, args):
    if not args.use_sam:
        return
    if args.sam_version == "sam1":
        model.load_sam("sam1", checkpoint_path=args.sam_checkpoint,
                       model_type=args.sam_model_type, k_pos=args.sam_k_pos,
                       k_neg=args.sam_k_neg, min_spacing_px=args.sam_spacing,
                       dilation_kernel=args.sam_dilation)
    else:
        model.load_sam("sam3", model_id=args.sam3_model_id, k_pos=args.sam_k_pos,
                       k_neg=args.sam_k_neg, min_spacing_px=args.sam_spacing,
                       dilation_kernel=args.sam_dilation)
    print(f"SAM{args.sam_version[-1]} refiner loaded.")


def run_dataset(classes, dataset_cls, split_val, data_path, model, batch_size):
    cat_maps, cat_gt = {}, {}
    for category in classes:
        print(f"\n  [{category}]")
        try:
            dataset = dataset_cls(
                source=data_path,
                classname=category,
                split=split_val,
                resize=model.img_resize,
                imagesize=model.img_resize,
                clip_transformer=model.preprocess,
            )
        except Exception as e:
            print(f"    Skipping: {e}")
            continue
        if len(dataset) == 0:
            print("    No images.")
            continue

        anomaly_maps, image_paths, gt_masks = model.infer(
            dataset, batch_size=batch_size, with_masks=True
        )
        if gt_masks is None or gt_masks.sum() == 0:
            print("    No GT mask pixels — skipping.")
            continue

        anomaly_maps = model.refine_with_sam(anomaly_maps, image_paths)
        cat_maps[category] = anomaly_maps
        cat_gt[category]   = gt_masks.astype(np.int32)

        del dataset, gt_masks
        gc.collect()
        torch.cuda.empty_cache()

    return cat_maps, cat_gt


def report_segf1(label, classes, cat_maps, cat_gt, threshold):
    print(f"\n{'='*90}")
    print(f"  {label}  |  threshold={threshold:.6f}")
    print(f"{'='*90}")
    print(f"  {'Category':16s}  {'SegF1':>7s}  {'AUROC-px':>8s}  {'AUROC-im':>8s}"
          f"  {'AP-px':>7s}  {'AP-im':>7s}  {'AUPRO':>7s}  {'F1-px':>7s}")
    print(f"  {'-'*84}")

    scores = []
    metric_cols = {k: [] for k in ["auroc_px", "auroc_sp", "ap_px", "ap_sp", "aupro", "f1_px"]}

    for cat in classes:
        if cat not in cat_maps:
            print(f"  {'  '+cat:16s}  (skipped)")
            continue
        pr_px = cat_maps[cat]
        gt_px = cat_gt[cat]

        segf1 = compute_segf1_at_threshold(gt_px, pr_px, threshold)
        scores.append(segf1)

        pr_flat = pr_px.reshape(pr_px.shape[0], -1)
        gt_flat = gt_px.reshape(gt_px.shape[0], -1)
        pr_sp = pr_flat.max(axis=1)
        gt_sp = (gt_flat.sum(axis=1) > 0).astype(np.int32)

        (auroc_sp, f1_sp, ap_sp), (auroc_px, f1_px, ap_px, aupro) = compute_metrics(
            gt_sp=gt_sp, pr_sp=pr_sp, gt_px=gt_px, pr_px=pr_px
        )
        for k, v in zip(["auroc_px", "auroc_sp", "ap_px", "ap_sp", "aupro", "f1_px"],
                        [auroc_px, auroc_sp, ap_px, ap_sp, aupro, f1_px]):
            metric_cols[k].append(v)

        print(f"  {cat:16s}  {segf1*100:>6.2f}%  {auroc_px*100:>7.2f}%  {auroc_sp*100:>7.2f}%"
              f"  {ap_px*100:>6.2f}%  {ap_sp*100:>6.2f}%  {aupro*100:>6.2f}%  {f1_px*100:>6.2f}%")

    if scores:
        mean = sum(scores) / len(scores)
        means = {k: sum(v) / len(v) for k, v in metric_cols.items() if v}
        print(f"  {'-'*84}")
        print(f"  {'mean':16s}  {mean*100:>6.2f}%  {means['auroc_px']*100:>7.2f}%"
              f"  {means['auroc_sp']*100:>7.2f}%  {means['ap_px']*100:>6.2f}%"
              f"  {means['ap_sp']*100:>6.2f}%  {means['aupro']*100:>6.2f}%"
              f"  {means['f1_px']*100:>6.2f}%")
        return mean
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold",      type=float, required=True)
    parser.add_argument("--backbone_name",  default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--mvtec1_path",    default="./data/mvtec_anomaly_detection/")
    parser.add_argument("--visa_path",      default="./data/visa/")
    parser.add_argument("--mvtec2_path",    default="./data/mvtec_ad_2/")
    parser.add_argument("--img_resize",     type=int, default=512)
    parser.add_argument("--feature_layers", type=int, nargs="+", default=[5, 11, 17, 23])
    parser.add_argument("--r_list",         type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--batch_size",     type=int, default=4)
    parser.add_argument("--device",         type=int, default=0)
    parser.add_argument("--datasets",       nargs="+", default=["mvtec2"],
                        choices=["mvtec1", "visa", "mvtec2"])
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
    print(f"Threshold : {args.threshold}")
    print(f"Datasets  : {args.datasets}")

    model = EthMuSAM(
        backbone_name=args.backbone_name,
        img_resize=args.img_resize,
        device=device,
        features_list=[l + 1 for l in args.feature_layers],
        r_list=args.r_list,
    )
    _build_sam(model, args)

    dataset_means = {}

    if "mvtec1" in args.datasets:
        import datasets.mvtec as mvtec
        print("\n" + "=" * 60)
        print("MVTec AD1 — inference")
        cat_maps, cat_gt = run_dataset(
            _MVTEC1_CLASSES, mvtec.MVTecDataset, mvtec.DatasetSplit.TEST,
            args.mvtec1_path, model, args.batch_size,
        )
        dataset_means["MVTec AD1"] = report_segf1(
            "MVTec AD1", _MVTEC1_CLASSES, cat_maps, cat_gt, args.threshold
        )

    if "visa" in args.datasets:
        import datasets.visa as visa
        print("\n" + "=" * 60)
        print("VisA — inference")
        cat_maps, cat_gt = run_dataset(
            _VISA_CLASSES, visa.VisaDataset, visa.DatasetSplit.TEST,
            args.visa_path, model, args.batch_size,
        )
        dataset_means["VisA"] = report_segf1(
            "VisA", _VISA_CLASSES, cat_maps, cat_gt, args.threshold
        )

    if "mvtec2" in args.datasets:
        import datasets.mvtec_ad2 as mvtec_ad2
        print("\n" + "=" * 60)
        print("MVTec AD2 — inference")
        cat_maps, cat_gt = run_dataset(
            _MVTEC2_CLASSES, mvtec_ad2.MVTecAD2Dataset, mvtec_ad2.DatasetSplit.TEST,
            args.mvtec2_path, model, args.batch_size,
        )
        dataset_means["MVTec AD2"] = report_segf1(
            "MVTec AD2", _MVTEC2_CLASSES, cat_maps, cat_gt, args.threshold
        )

    if dataset_means:
        print(f"\n{'='*55}")
        print(f"  SUMMARY  |  threshold={args.threshold:.6f}")
        print(f"{'='*55}")
        for ds, mean in dataset_means.items():
            print(f"  {ds:16s}  {mean*100:>7.2f}%")
        overall = sum(dataset_means.values()) / len(dataset_means)
        print(f"  {'-'*28}")
        print(f"  {'overall mean':16s}  {overall*100:>7.2f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
