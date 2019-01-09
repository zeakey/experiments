import torch
import torchvision
from models import msnet1
import time

msnet50 = msnet1.msnet50().cuda()
resnet50 = torchvision.models.resnet.resnet50().cuda()


benchmark = True
bs = 32
num_batch = 100
if benchmark:
    start_time = time.time()
    for i in range(num_batch):
        output = msnet50(torch.zeros(bs, 3, 224, 224).cuda())
    print("MSNet50: %.5f seconds per batch." % ((time.time() - start_time)/num_batch))
    
    start_time = time.time()
    for i in range(num_batch):
        output = resnet50(torch.zeros(bs, 3, 224, 224).cuda())
    print("ResNet50: %.5f seconds per batch." % ((time.time() - start_time)/num_batch))
