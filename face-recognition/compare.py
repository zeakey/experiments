from scipy.io import loadmat
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import sys, os, copy
from os.path import join, isfile, isdir

assert len(sys.argv) >= 3, "At least 2 targets (given %d)" % (len(sys.argv) - 1)

mats = sys.argv[1:]

for m in mats:
    assert m.endswith("record.mat")

legends = [m.replace("/record.mat", "") for m in mats]


print(mats)

plt.style.use('seaborn')
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i in range(len(mats)):

    data = loadmat(mats[i])

    axes[0].plot(data["train_loss_record"].squeeze())
    axes[0].set_title("Loss")

    axes[1].plot(data["train_exloss_record"].squeeze())
    axes[1].set_title("Ex Loss")

    axes[2].plot(data["train_acc1_record"].squeeze())
    axes[2].set_title("Train Acc1")

    axes[3].plot(data["lfw_acc_record"].squeeze())
    axes[3].set_title("LFW Acc")

    axes[3].legend(legends)

plt.savefig("compare.pdf")
