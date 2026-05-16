#!/bin/bash
# Generate MVTec AD2 submission using DINOv3 + SAM3 (EthMuSAM).
# Calibrate threshold first via scripts/find_threshold_mvtec1.py or
# scripts/find_threshold_visa.py, then set --threshold below.

python src/submit.py \
    --backbone_name facebook/dinov3-vitl16-pretrain-lvd1689m \
    --img_resize 512 \
    --feature_layers 5 11 17 23 \
    --r_list 1 3 5 \
    --batch_size 4 \
    --device 0 \
    --use_sam \
    --sam_version sam3 \
    --sam3_model_id models/sam3 \
    --threshold 0.357525
