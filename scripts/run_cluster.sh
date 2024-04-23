#!/bin/bash



python main.py --dataset "CLUSTER"\
               --gamma 0.3\
               --n_hop 0\
               --warmup_epoch 5\
               --epochs 100\
               --slope 0.1\
               --channels 48\
               --nhead 8\
               --dropout 0\
               --batch_size 16\
               --eval_batch_size 16\
               --num_layers 16\
               --pe_norm True\
               --pe_origin_dim 10\
               --pe_dim 16\
               --lr 0.0005\
               --seed 42\
               --run 10\
               --devices 0

