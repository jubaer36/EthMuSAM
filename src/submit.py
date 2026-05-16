"""Generate MVTec AD 2 submission files for benchmark.mvtec.com.

Runs EthMuSAM on test_private and test_private_mixed splits and writes:
  anomaly_images/{category}/{split}/{stem}.tiff        — float16 anomaly scores
  anomaly_images_thresholded/{category}/{split}/{stem}.png  — binary masks (0/255)

Upload the output folder to benchmark.mvtec.com after verifying with
MVTecAD2_public_code_utils/check_and_prepare_data_for_upload.py.

Usage:
    python src/submit.py --threshold 0.357525 --data_path ./data/mvtec_ad_2/
"""

import argparse
import gc
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.model import EthMuSAM, CLASSNAMES

import warnings
warnings.filterwarnings("ignore")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class PrivateSplitDataset(torch.utils.data.Dataset):
    """Loads images from test_private / test_private_mixed (no GT masks available)."""

    def __init__(self, data_path, category, split, image_size=512, transform=None):
        self.split_dir = Path(data_path) / category / split
        self.image_paths = sorted(self.split_dir.glob("*.png"))
        if not self.image_paths:
            raise FileNotFoundError(f"No PNG images in {self.split_dir}")
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = str(self.image_paths[idx])
        return {"image": self.transform(Image.open(path).convert("RGB")), "image_path": path}


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
    parser = argparse.ArgumentParser(description="EthMuSAM — generate MVTec AD 2 submission")
    parser.add_argument("--data_path",      default="./data/mvtec_ad_2/")
    parser.add_argument("--classes",        nargs="+", default=CLASSNAMES)
    parser.add_argument("--backbone_name",  default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--img_resize",     type=int, default=512)
    parser.add_argument("--feature_layers", type=int, nargs="+", default=[5, 11, 17, 23])
    parser.add_argument("--r_list",         type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--batch_size",     type=int, default=4)
    parser.add_argument("--device",         type=int, default=0)
    parser.add_argument("--threshold",      type=float, required=True,
                        help="Fixed threshold calibrated on an independent dataset.")
    parser.add_argument("--submission_dir", default=None,
                        help="Output directory (default: ./{backbone_short}_submission_folder).")
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

    model = EthMuSAM(
        backbone_name=args.backbone_name,
        img_resize=args.img_resize,
        device=device,
        features_list=[l + 1 for l in args.feature_layers],
        r_list=args.r_list,
    )
    _build_sam(model, args)

    submission_dir = Path(args.submission_dir or f"./{model.short_name}_submission_folder")

    print(f"Backbone       : {args.backbone_name}")
    print(f"Submission dir : {submission_dir}")
    print(f"Threshold      : {args.threshold}")
    print(f"Device         : {device}")

    for category in args.classes:
        for split in ["test_private", "test_private_mixed"]:
            print(f"\n[{category}] {split}")
            try:
                dataset = PrivateSplitDataset(
                    data_path=args.data_path,
                    category=category,
                    split=split,
                    image_size=args.img_resize,
                    transform=model.preprocess,
                )
            except FileNotFoundError as e:
                print(f"  Skipping: {e}")
                continue

            print(f"  {len(dataset)} images")
            anomaly_maps, image_paths = model.infer(
                dataset, batch_size=args.batch_size, with_masks=False
            )
            anomaly_maps = model.refine_with_sam(anomaly_maps, image_paths)
            model.save_submission(
                anomaly_maps, image_paths, category, split, submission_dir, args.threshold
            )

            del anomaly_maps, image_paths, dataset
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\nSubmission folder: {submission_dir.resolve()}")
    print("\nVerify and package for upload:")
    print(f"  cd MVTecAD2_public_code_utils && \\")
    print(f"  python check_and_prepare_data_for_upload.py ../{submission_dir}")


if __name__ == "__main__":
    main()
