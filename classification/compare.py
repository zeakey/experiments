from scipy.io import loadmat
import matplotlib.pyplot as plt
mats = ["tmp/cifar100-resnet_se_cifar.resnet18/record.mat",
        "tmp/cifar100-resnet_prelufc_cifar.resnet18/record.mat",
        "tmp/cifar100-resnet_cifar.resnet18/record.mat"]



for i in range(len(mats)):
    mats[i] = loadmat(mats[i])
    print(mats[i]["acc1"].squeeze().shape)
    plt.plot(mats[i]["acc1"].squeeze())
plt.grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
plt.legend(["SE-ResNet", "PReLU-fc", "ResNet"])
plt.savefig("se-vs-prelufc-vs-resnet18.pdf")