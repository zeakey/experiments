for SETTING in 0 1 2 3
do
    for RUN in 1 2 3
    do
        python -m torch.distributed.launch --nproc_per_node=4 train.py --train-data data/MSRA10K --fp16 --batch-size 16 \
        --tmp "tmp/drn_d_22_baseline-imagenet-pretrain-setting"$SETTING"-run"$RUN --pretrained drn_d_22-4bd2f8ea.pth --arch drn_d_22 --setting $SETTING
    done
done