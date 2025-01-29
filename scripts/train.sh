#!/bin/bash

python train.py \
    --source_data data/office31/amazon \
    --target_data data/office31/webcam \
    --dataset_type office31 \
    --num_classes 31 \
    --use_rbn \
    --replace_layer 3 \
    --epochs 50 \
    --batch_size 32 \
    --output_dir output/office31_a2w
