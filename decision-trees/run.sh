set -ex
for t in 4 5 6
do
    for d in  5 6 7 8 9
    do
      CUDA_VISIBLE_DEVICES=1 python cifar.py --use-forest --tmp tmp/forest-t"$t"d"$d" --num-trees $t --tree-depth $d --num-classes 100
    done
done
