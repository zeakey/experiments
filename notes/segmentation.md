## 2020-May-5
torchvision official code, `python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py  -b 8 --model fcn_resnet50 --aux-los`
```
Test: Total time: 0:00:49
global correct: 89.8
average row correct: ['96.6', '80.2', '71.3', '69.8', '52.8', '60.7', '77.1', '75.9', '84.7', '32.6', '64.2', '43.7', '68.3', '68.0', '79.7', '88.0', '51.6', '75.2', '42.2', '71.6', '66.8']
IoU: ['91.0', '73.0', '52.0', '58.4', '48.1', '50.4', '68.9', '68.9', '66.7', '21.3', '52.4', '37.2', '56.3', '50.8', '64.8', '75.4', '44.5', '57.1', '31.4', '62.4', '52.7']
mean IoU: 56.4
Training time 2:38:36
```
