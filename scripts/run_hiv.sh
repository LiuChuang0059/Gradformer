#!/bin/bash



python main.py --dataset "ogbg-molhiv"\
               --gamma 0.7\
               --n_hop 5\
               --warmup_epoch 5\
               --slope 0.2\
               --epochs 100\
               --channels 64\
               --nhead 4\
               --dropout 0.05\
               --batch_size 32\
               --eval_batch_size 32\
               --num_layers 10\
               --pe_norm True\
               --pe_origin_dim 16\
               --pe_dim 16\
               --lr 0.0001\
               --run 10\
               --seed 42\
               --devices 0

