#!/bin/bash

# arg1: GPU ID
# arg2: depth [20, 32]
# arg3: prune-rate

if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
    echo "arg1: GPU ID, arg2: network depth, arg3: prune-rate"
    exit
fi

MODEL="cifar_resnet"$2"_v1"
TMP="tmp/cifar/"$MODEL"-"$3

echo "Model name: "$MODEL
echo "tmp folder: "$TMP

for VARIABLE in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=$1 python finetune_cifar.py --mode hybrid --prune-grad --model $MODEL --prune-rate 0.3 --save-dir $TMP
done