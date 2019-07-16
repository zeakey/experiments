#!/bin/bash

CUDAVISIBLE_DEVICES=$1 python train_cifar10.py --model cifar_resnet56_v2 \
--lr 0.01 --lr-decay 10,0.2,0.2,0.2 --lr-decay-epoch 1,60,120,160 \
--pruning-rate 0.4 --save-dir tmp/cifar_resnet56_v2_pruning0.4

# hyper-parameters strictly follow
# https://github.com/he-y/soft-filter-pruning/blob/master/scripts/cifar10_resnet.sh
