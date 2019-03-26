import torch
import torch.utils.data as data
from torchvision.datasets.folder import pil_loader
import os
from os.path import join, split, splitext, abspath, dirname, isfile, isdir
from scipy.io import loadmat
from datetime import datetime, timedelta
import random

random.seed(7)

from vltools.image import isimg

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


class UTKFaceDataset(torch.utils.data.Dataset):

    def __init__(self, root, split="train", transforms=None):

        self.root = root
        self.transforms = transforms
        assert isdir(root)
        self.images = [i for i in os.listdir(self.root) if isimg(join(self.root, i))]
        random.shuffle(self.images)

        split_point = (len(self.images) // 10) * 9
        if split == "train":
            self.images = self.images[0:split_point]
        elif split == "test":
            self.images = self.images[split_point::]
        else:
            raise ValueError("Invalid split %s" % split)

        assert len(self.images) > 0

    def __getitem__(self, index):

        i = self.images[index]
        im = pil_loader(join(self.root, i))
        try:
            age, gender, race, _ = splitext(i)[0].split("_")
            age = int(age)
            gender = int(gender)
            race = int(race)
        except:
            print(i)

        if self.transforms:
            im = self.transforms(im)

        return im, age, gender, race

    def __len__(self):
        return len(self.images)

if __name__ == "__main__":

    # imdb = IMDBDataset("/media/data2/dataset/IMDB-face/imdb_crop", "/media/data2/dataset/IMDB-face/imdb/imdb.mat")
    utk = UTKFaceDataset("/media/data2/dataset/UTKFace-aligned")
    print(utk.__len__())
    for im, age, gender, race in utk:
        print(im, age, gender, race)
