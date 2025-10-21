
#!/bin/bash
# Activate the virtual environment

source /home/jupyter/palaskasc/Transformer_distillation/CHAD/bin/activate

torchrun --standalone --nproc_per_node=8 --master_port=29500 main.py --dist_url env://  --distributed --data-path /home/jupyter/imagenet --teacher-path /home/jupyter/palaskasc/Transformer_distillation/S24_224.pth --batch-size 64 --model deit_tiny_patch16_224 --teacher-model cait_s24_224 --distillation-type soft --distillation-alpha 1 --distillation-beta 0.3 --gamma 3 --delta 0 --K 192 --s-id 11 --t-id 23 --drop-path 0 --output_dir ./cait_s24_224_100_epochs_single_loss --num_workers 12 --epochs 100 --normalize --distance KL --use-prototypes --prototypes-number 1024
torchrun --standalone --nproc_per_node=8 --master_port=29500 main.py --dist_url env://  --distributed --data-path /home/jupyter/imagenet --teacher-path /home/jupyter/palaskasc/Transformer_distillation/S24_224.pth --batch-size 64 --model deit_tiny_patch16_224 --teacher-model cait_s24_224 --distillation-type soft --distillation-alpha 1 --distillation-beta 0.3 --gamma 0 --delta 1 --K 192 --s-id 11 --t-id 23 --drop-path 0 --output_dir ./cait_s24_224_100_epochs_single_loss --num_workers 12 --epochs 100 --normalize --distance KL --use-prototypes --prototypes-number 1024
