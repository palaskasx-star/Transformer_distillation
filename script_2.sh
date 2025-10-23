#!/bin/bash
source /home/jupyter/palaskasc/Transformer_distillation/CHAD/bin/activate

cd ../manifold-distillation
torchrun --standalone --nproc_per_node=8 --master_port=29500 main.py --dist_url env:// --distributed --output_dir ../Transformer_distillation/cait_s24_224_100_epochs_single_loss/manifold_distillation_last_layers --data-path /home/jupyter/imagenet --teacher-path /home/jupyter/palaskasc/Transformer_distillation/S24_224.pth --model deit_tiny_patch16_224 --teacher-model cait_s24_224 --distillation-type soft --epochs 100 --distillation-alpha 1 --distillation-beta 1 --w-sample 0.1 --w-patch 4 --w-rand 0.2 --K 192 --s-id 11 --t-id 23 --drop-path 0 --resume /home/jupyter/palaskasc/Transformer_distillation/cait_s24_224_100_epochs_single_loss/manifold_distillation_last_layers/checkpoint.pth

cd home/jupyter/palaskasc/Transformer_distillation


torchrun --standalone --nproc_per_node=8 --master_port=29500 main.py --dist_url env://  --distributed --data-path /home/jupyter/imagenet --teacher-path /home/jupyter/palaskasc/Transformer_distillation/S24_224.pth --batch-size 64 --model deit_tiny_patch16_224 --teacher-model cait_s24_224 --distillation-type soft --distillation-alpha 1 --distillation-beta 1 --gamma 3 --delta 0 --K 192 --s-id 11 --t-id 23 --drop-path 0 --output_dir ./cait_s24_224_100_epochs_single_loss_different_prots_projs --num_workers 12 --epochs 100 --normalize --distance KL --use-prototypes --prototypes-number 1024 --resume /home/jupyter/palaskasc/Transformer_distillation/cait_s24_224_100_epochs_single_loss_different_prots_projs/normalize_True_distance_KL_beta_1.0_gamma_3.0_delta_0.0_K_192_sids_11_tids_23_prototypes_1024/checkpoint.pth
