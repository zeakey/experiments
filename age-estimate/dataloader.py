import torch
import torch.utils.data as data
from torchvision.datasets.folder import pil_loader
from os.path import join, split, splitext, abspath, dirname, isfile, isdir
from scipy.io import loadmat

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

        return im, age

    def __len__(self):
        return len(self.items)

class IMDBDataset(torch.utils.data.Dataset):

    def __init__(self, root, metadata, transforms):

        self.root = root
        self.transforms = transforms
        assert isfile(metadata)
        self.metadata = loadmat(metadata)


    def __getitem__(self, index):

        filename, age = self.items[index].split()
        age = int(age)
        im = pil_loader(join(self.root, filename))

        if self.transforms:
            im = self.transforms(im)

        return im, age

    def __len__(self):
        return len(self.items)

if __name__ == "__main__":

    imdb = IMDBDataset("/media/data2/IMDB-")
