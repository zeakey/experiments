import torch
import torch.utils.data as data
from torchvision.datasets.folder import pil_loader
from os.path import join, split, splitext, abspath, dirname, isfile, isdir
from scipy.io import loadmat
from datetime import datetime, timedelta

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

    def __init__(self, root, metadata, transforms=None):

        self.root = root
        self.transforms = transforms
        assert isfile(metadata)
        self.metadata = loadmat(metadata)["imdb"]

        self.birth_day = self.metadata[0][0][0]
        self.photo_taken_date = self.metadata[0][0][1]
        self.items = self.metadata[0][0][2]
        self.gender = self.metadata[0][0][3]

        assert self.items.size == self.gender.size


    def __getitem__(self, index):

        filename = str(self.items[0, index][0])
        im = pil_loader(join(self.root, filename))
        birth_day = self.matlab_datenum_to_pydate(self.birth_day[0, index]).year
        age = self.photo_taken_date[0, index] - birth_day

        if self.transforms:
            im = self.transforms(im)

        return im, int(self.gender[0, index])

    @staticmethod
    def matlab_datenum_to_pydate(matlab_datenum):

        matlab_datenum = int(matlab_datenum)

        return datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum%1) - timedelta(days = 366)

    def __len__(self):

        return self.items.size

if __name__ == "__main__":

    imdb = IMDBDataset("/media/data2/dataset/IMDB-face/imdb_crop", "/media/data2/dataset/IMDB-face/imdb/imdb.mat")
