#!/bin/bash

# Office-31 数据集测试脚本
python test.py \
    --test_data data/office31/webcam \
    --dataset_type office31 \
    --num_classes 31 \
    --model_path output/office31_a2w/model_final.pth \
    --use_rbn \
    --replace_layer 3 \
    --batch_size 32
