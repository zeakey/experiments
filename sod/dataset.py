import torch
from PIL import Image
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from os.path import join, isdir, isfile
from skimage.filters import sobel_h, sobel_v
import cv2
from vlkit.dense import seg2edge, dense2flux, flux2angle, quantize_angle, dequantize_angle

def edge2flux(edge):
    H, W = edge.shape

    yy = np.arange(H).reshape(H, 1)
    yy = np.repeat(yy, W, axis=1).reshape(H, W, 1)

    xx = np.arange(W).reshape(1, W)
    xx = np.repeat(xx, H, axis=0).reshape(H, W, 1)

    yyxx = np.concatenate((yy, xx), axis=2).transpose((2, 0, 1))

    dist, inds = distance_transform_edt(np.logical_not(edge), return_indices=True)

    field = inds - yyxx

    # normalize
    norm = np.sqrt(np.sum(field**2, axis=0))
    norm[norm==0] = 1
    field = (field / norm).astype(np.float32)

    return field, dist.astype(np.float32)


class SODDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, name_list, flip=True, image_transform=None, label_transform=None):
        assert isdir(img_dir), "%s doesn't exist" % img_dir
        assert isdir(label_dir), "%s doesn't exist" % label_dir
        assert isfile(name_list)

        self.img_dir = img_dir
        self.label_dir = label_dir
        self.flip = flip
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.__item_names = [line.strip() for line in open(name_list, 'r')]

        assert isdir(img_dir) and isdir(label_dir)

    def __getitem__(self, index):        
        img_fullname = join(self.img_dir, self.__item_names[index]+".jpg")
        assert isfile(img_fullname), img_fullname
        image = Image.open(img_fullname).convert("RGB")

        lb_fullname = join(self.label_dir, self.__item_names[index]+".png")
        assert isfile(lb_fullname), lb_fullname
        label = Image.open(lb_fullname).convert("L")

        flip = self.flip and np.random.rand() > 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        if self.image_transform:
            image = self.image_transform(image)
        if self.label_transform:
            label = self.label_transform(label)

        label = np.array(label).astype(bool)
        
        edge = seg2edge(label)
        dist = distance_transform_edt(np.logical_not(edge))
        mask = dist <= 15
        mask[dist <= 1] = False

        angle = flux2angle(dense2flux(dist))
        orientation = quantize_angle(angle, num_bins=8)

        label = np.expand_dims(label, axis=0)
        edge = np.expand_dims(edge, axis=0)
        # orientation = np.expand_dims(orientation, axis=0)
        # mask = np.expand_dims(mask, axis=0)
        metas = dict({
            "filename": img_fullname,
            "flip": flip
        })
        result = dict({
            "image": image,
            "label": label,
            "edge": edge,
            "orientation": orientation,
            "mask": mask,
            "metas": metas
        })

        return result

    def __len__(self):
        return len(self.__item_names)

    def get_item_names(self):
        return self.__item_names.copy()
