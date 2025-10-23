#!/bin/bash
source /home/jupyter/palaskasc/Transformer_distillation/CHAD/bin/activate

cd ../manifold-distillation
torchrun --standalone --nproc_per_node=8 --master_port=29500 main.py --dist_url env:// --distributed --output_dir ../Transformer_distillation/cait_s24_224_100_epochs_single_loss/manifold_distillation_last_layers --data-path /home/jupyter/imagenet --teacher-path /home/jupyter/palaskasc/Transformer_distillation/S24_224.pth --model deit_tiny_patch16_224 --teacher-model cait_s24_224 --distillation-type soft --epochs 100 --distillation-alpha 1 --distillation-beta 1 --w-sample 0.1 --w-patch 4 --w-rand 0.2 --K 192 --s-id 11 --t-id 23 --drop-path 0 --resume /home/jupyter/palaskasc/Transformer_distillation/cait_s24_224_100_epochs_single_loss/manifold_distillation_last_layers/checkpoint.pth

cd home/jupyter/palaskasc/Transformer_distillation
