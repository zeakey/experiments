import torch
from torchvision.datasets.folder import pil_loader
from os.path import isfile, join

class LFWDataset(torch.utils.data.Dataset):

    def __init__(self, root, filelist, transforms=None):

        self.root = root
        self.transforms = transforms
        assert isfile(filelist), filelist

        with open(filelist) as f:
            self.items = f.readlines()

        for i in range(self.__len__()):
            self.items[i] = self.items[i].strip()

    def __getitem__(self, index):

        filename = self.items[index]
        assert isfile(join(self.root, filename))
        im = pil_loader(join(self.root, filename))

        if self.transforms:
            im = self.transforms(im)

        return im

    def __len__(self):
        return len(self.items)
