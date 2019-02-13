import torch
import torch.utils.data as data
from torchvision.datasets.folder import pil_loader
from os.path import join, split, splitext, abspath, dirname, isfile, isdir

class CACDDataset(torch.utils.data.Dataset):

    def __init__(self, root, filelist, transforms=None):

        self.root = root
        self.transforms = transforms
        assert isfile(filelist)

        # list files: `cacd_train_list.txt`, `cacd_test_list.txt`, `cacd_val_list.txt`
        with open(filelist) as f:
            self.items = f.readlines()

    def __getitem__(self, index):

        filename, age = self.items[index].split()
        age = int(age)
        im = pil_loader(join(self.root, filename))

        if self.transforms:
            im = self.transforms(im)
        assert age >= 14 and age <= 62, "age=%d" % age
        age = age - 14

        return im, age

    def __len__(self):
        return len(self.items)