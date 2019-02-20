from scipy.io import loadmat
import matplotlib.pyplot as plt
mats = ["tmp/cifar100-resnet_se_cifar.resnet34/record.mat",
        "tmp/cifar100-resnet_prelufc_cifar.resnet34/record.mat",
        "tmp/cifar100-resnet_cifar.resnet34/record.mat"]

legends = ["SE-ResNet", "PReLU-fc", "ResNet"]

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for i in range(len(mats)):
    mats[i] = loadmat(mats[i])
    axes[0].plot(mats[i]["acc1"].squeeze())
    axes[0].plot(mats[i]["acc5"].squeeze())
    axes[0].set_title("Acc")

    axes[1].plot(mats[i]["loss_record"].squeeze())
    axes[1].set_title("Loss")


for ax in axes:
    ax.grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
    ax.legend(legends)

# plt.legend(["SE-ResNet", "PReLU-fc", "ResNet"])
plt.savefig("se-vs-prelufc-vs-resnet34.pdf")