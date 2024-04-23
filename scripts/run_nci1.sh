#!/bin/bash


for gamma in 0.6 0.7
do
          python main.py --dataset "NCI1"\
                         --gamma $gamma\
                         --n_hop 5\
                         --epochs 100\
                         --slope 0.0\
                         --batch_size 128\
                         --eval_batch_size 128\
                         --num_layers 4\
                         --run 10\
                         --devices 0
done


