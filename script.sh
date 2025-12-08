#!/bin/bash
# Activate the virtual environment

source /home/jupyter/palaskasc/Transformer_distillation/CHAD/bin/activate

torchrun --standalone --nproc_per_node=4 --master_port=29500 main.py --dist_url env://  --distributed --data-path /home/jupyter/imagenet --teacher-path /home/jupyter/palaskasc/Transformer_distillation/S24_224.pth --batch-size 128 --model deit_tiny_patch16_224 --teacher-model cait_s24_224 --distillation-type soft --distillation-alpha 1 --distillation-beta 0.3 --w-sample 0.1 --w-patch 4 --w-rand 0.2 --K 192 --s-id 11 --t-id 23 --drop-path 0 --output_dir ./report --num_workers 12 --epochs 300 --normalize --distance KL --use-prototypes --prototypes-number 1024


