#!/bin/bash
# Activate the virtual environment

source /home/jupyter/palaskasc/Transformer_distillation/CHAD/bin/activate

cd ..
cd manifold-distillation

torchrun --standalone --nproc_per_node=8 --master_port=29500 main.py --dist_url env:// --distributed --output_dir ./manifold_distillation --data-path /home/jupyter/imagenet --teacher-path /home/jupyter/palaskasc/Transformer_distillation/S24_224.pth --model deit_tiny_patch16_224 --teacher-model cait_s24_224 --distillation-type soft --distillation-alpha 1 --distillation-beta 1 --w-sample 0.1 --w-patch 4 --w-rand 0.2 --K 192 --s-id 0 1 2 3 8 9 10 11 --t-id 0 1 2 3 20 21 22 23 --drop-path 0 
