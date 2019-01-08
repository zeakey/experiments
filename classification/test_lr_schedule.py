import torch, torchvision
from utils import CosineAnnealingLR
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import utils
from models import msnet
model = msnet.MSNet50()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
print(model)
print(optimizer)

with torch.no_grad():
    data = torch.zeros((1, 3, 224, 224))
    output = model(data)
print(output.mean())

# lr_scheduler0 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0)
# lr_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

lr_scheduler = CosineAnnealingLR(optimizer, T_max=120, eta_min=0, restart_every=40, restart_factor=0.8)

lr_record = []

# for i in range(120):
#     lr_scheduler.step()

#     lr_record.append(optimizer.param_groups[0]["lr"])

# plt.plot(lr_record)
# plt.savefig("lr_scheduler.pdf")


for i in range(120):
    lr_record.append(utils.get_lr(i, 0.1, warmup_epochs=5, warmup_start_lr=0.001))

plt.plot(lr_record)
plt.savefig("lr_scheduler.pdf")

plt.clf()
lr_record = []
for i in range(120):
    lr_record.append(utils.get_lr(i, 0.1, warmup_epochs=0))

plt.plot(lr_record)
plt.savefig("lr_scheduler1.pdf")


plt.clf()
lr_record = []
for i in range(120):
    for j in range(1000):
        lr_record.append(utils.get_lr_per_iter(i,j, iters_per_epoch=1000, base_lr=0.1, stepsize=30, warmup_epochs=5))

plt.plot(lr_record)
plt.savefig("lr_scheduler2.pdf")
