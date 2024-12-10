#!/bin/bash

for data in ZINC
do
for layer in 10
do
    for gamma in 0.6
		do
		for hop in 2
		do
		  for run in 0 1 2 3 4 5 6 7 8 9
		          do
                python main.py --dataset $data\
                         --gamma $gamma\
                         --n_hop $hop\
                         --ablation 0\
                         --warmup_epoch 50\
                         --slope 0.1\
                         --epochs 2000\
                         --channels 64\
                         --nhead 4\
                         --dropout 0\
                         --batch_size 32\
                         --eval_batch_size 32\
                         --num_layers $layer\
                         --pe_norm True\
                         --pe_origin_dim 20\
                         --pe_dim 28\
                         --lr 0.001\
                         --run $run\
                         --devices 0
               done
		done
		done
	done
done
