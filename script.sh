#!/bin/bash
# Activate the virtual environment

source /home/jupyter/palaskasc/Transformer_distillation/CHAD/bin/activate

torchrun --standalone --nproc_per_node=8 --master_port=29500 main.py --dist_url env://  --distributed --data-path /home/jupyter/imagenet --teacher-path /home/jupyter/palaskasc/Transformer_distillation/S24_224.pth --batch-size 64 --model deit_tiny_patch16_224 --teacher-model cait_s24_224 --distillation-type soft --distillation-alpha 1 --distillation-beta 2 --gamma 1 --delta 0.1 --K 192 --s-id 11 --t-id 23 --drop-path 0 --output_dir ./cait_s24_224_100_epochs_intra_rand_search --num_workers 12 --epochs 100 --normalize --distance KL --use-prototypes --prototypes-number 1024 --resume /home/jupyter/palaskasc/Transformer_distillation/cait_s24_224_100_epochs_intra_rand_search/normalize_True_distance_KL_beta_2.0_gamma_1.0_delta_0.1_K_192_sids_11_tids_23_prototypes_1024/checkpoint.pth
torchrun --standalone --nproc_per_node=8 --master_port=29500 main.py --dist_url env://  --distributed --data-path /home/jupyter/imagenet --teacher-path /home/jupyter/palaskasc/Transformer_distillation/S24_224.pth --batch-size 64 --model deit_tiny_patch16_224 --teacher-model cait_s24_224 --distillation-type soft --distillation-alpha 1 --distillation-beta 2 --gamma 1 --delta 0.4 --K 1024 --s-id 11 --t-id 23 --drop-path 0 --output_dir ./cait_s24_224_100_epochs_intra_rand_search --num_workers 12 --epochs 100 --normalize --distance KL --use-prototypes --prototypes-number 1024
torchrun --standalone --nproc_per_node=8 --master_port=29500 main.py --dist_url env://  --distributed --data-path /home/jupyter/imagenet --teacher-path /home/jupyter/palaskasc/Transformer_distillation/S24_224.pth --batch-size 64 --model deit_tiny_patch16_224 --teacher-model cait_s24_224 --distillation-type soft --distillation-alpha 1 --distillation-beta 2 --gamma 1 --delta 0.25 --K 1024 --s-id 11 --t-id 23 --drop-path 0 --output_dir ./cait_s24_224_100_epochs_intra_rand_search --num_workers 12 --epochs 100 --normalize --distance KL --use-prototypes --prototypes-number 1024





