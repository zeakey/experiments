import torch
from PIL import Image
import numpy as np
from os.path import join, isdir, isfile

class SODDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dirs, name_list, flip=False, image_transform=None, label_transform=None):
        assert isdir(img_dir), "%s doesn't exist" % img_dir
        for label_dir in label_dirs:
            assert isdir(label_dir), "%s doesn't exist" % label_dir
        assert isfile(name_list)

        self.img_dir = img_dir
        self.label_dirs = label_dirs
        self.flip = flip
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.item_names = [line.strip() for line in open(name_list, 'r')]

    def __getitem__(self, index):
        assert isfile(join(self.img_dir, self.item_names[index]+".jpg"))
        img = Image.open(join(self.img_dir, self.item_names[index]+".jpg"))

        labels = []
        for label_dir in self.label_dirs:
            assert isfile(join(label_dir, self.item_names[index]+".png"))
            labels.append(Image.open(join(label_dir, self.item_names[index]+".png")))

        if self.flip and np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            labels = [label.transpose(Image.FLIP_LEFT_RIGHT) for label in labels]

        if self.image_transform:
            img = self.image_transform(img)
        if self.label_transform:
            labels = [self.label_transform(l) for l in labels]

        return tuple([img]+labels+[self.item_names[index]])

        # data = [Image.open(join(self.data_dir, self.image_list[index]))]
        # if self.GTlabel_list is not None:
        #     data.append(Image.open(join(self.data_dir, self.GTlabel_list[index])))

        # pseudo_labels=[]
        # if self.pseudolabel_list is not None:
        #     for single_label_list in self.pseudolabel_list:
        #         pseudo_labels.append(Image.open(join(self.data_dir, single_label_list[index])))
        # data.append(pseudo_labels)

        # data = list(self.transforms(*data))

        # if self.out_name:
        #     data.append(self.image_list[index])
        # assert len(data)==4

        # return tuple(data)

    def __len__(self):
        return len(self.item_names)