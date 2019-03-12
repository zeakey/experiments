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
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i in range(len(mats)):
    data = loadmat(mats[i])
    axes[0].plot(data["acc1"].squeeze())
    axes[0].set_title("Acc-1")

    axes[1].plot(data["acc5"].squeeze())
    axes[1].set_title("Acc-5")

    axes[2].plot(data["loss_record"].squeeze())
    axes[2].set_title("Loss")
    axes[2].legend(legends, fontsize=6)

plt.savefig("compare.pdf")
