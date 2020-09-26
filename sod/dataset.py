import torch
from PIL import Image
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from os.path import join, isdir, isfile
from skimage.filters import sobel_h, sobel_v
import cv2
from vlkit.dense import seg2edge, dense2flux, flux2angle, quantize_angle, dequantize_angle
from vlkit.io import imread

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
    def __init__(self, img_dir, label_dir, name_list, flip=True, imsize=[0,0], image_transform=None, label_transform=None):
        assert isdir(img_dir), "%s doesn't exist" % img_dir
        assert isdir(label_dir), "%s doesn't exist" % label_dir
        assert isfile(name_list)
        assert isinstance(imsize, list) and len(imsize) == 2

        self.img_dir = img_dir
        self.label_dir = label_dir
        self.flip = flip
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.imsize = np.array(imsize)
        self.__item_names = [line.strip() for line in open(name_list, 'r')]

        assert isdir(img_dir) and isdir(label_dir)

    def __getitem__(self, index):        
        img_fullname = join(self.img_dir, self.__item_names[index]+".jpg")
        assert isfile(img_fullname), img_fullname
        image = imread(img_fullname, backend="pil")

        lb_fullname = join(self.label_dir, self.__item_names[index]+".png")
        assert isfile(lb_fullname), lb_fullname
        label = imread(lb_fullname, backend="pil", grayscale=True)

        if image.shape[:2] != label.shape:
            print("image and label shape mismatch, reshape.")
            H, W = image.shape[:2]
            label = cv2.resize(label, (W, H))

        flip = self.flip and np.random.choice([True, False])
        if flip:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)

        if np.all(self.imsize > 0):
            H, W = self.imsize
            image = image.resize((W, H))
            label = label.resize((W, H))
        image = np.array(image, dtype=np.uint8)
        label = np.array(label, dtype=np.uint8)

        edge = seg2edge(label.astype(bool))
        dist = distance_transform_edt(np.logical_not(edge))
        mask = dist <= 15
        mask[dist <= 1] = False

        angle = flux2angle(dense2flux(dist))
        orientation = quantize_angle(angle, num_bins=8).astype(np.int64)

        if self.image_transform:
            image = self.image_transform(image)
        if self.label_transform:
            label = self.label_transform(label)
            edge = self.label_transform(edge)
            orientation = self.label_transform(orientation)
            mask = self.label_transform(mask)

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

# Caffe style data loading code
# image encode/decode with cv2
# images are not normalized (/255)
class CaffeDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, name_list, flip=True, imsize=[0,0]):
        assert isdir(img_dir), "%s doesn't exist" % img_dir
        assert isdir(label_dir), "%s doesn't exist" % label_dir
        assert isfile(name_list)
        assert isinstance(imsize, list) and len(imsize) == 2

        self.img_dir = img_dir
        self.label_dir = label_dir
        self.flip = flip
        self.imsize = np.array(imsize)
        self.__item_names = [line.strip() for line in open(name_list, 'r')]

        assert isdir(img_dir) and isdir(label_dir)

    def __getitem__(self, index):        
        img_fullname = join(self.img_dir, self.__item_names[index]+".jpg")
        assert isfile(img_fullname), img_fullname
        image = imread(img_fullname, backend="cv2")
        assert image.ndim == 3
        image = image - np.array((104.00699, 116.66877, 122.67892))

        lb_fullname = join(self.label_dir, self.__item_names[index]+".png")
        assert isfile(lb_fullname), lb_fullname
        label = imread(lb_fullname, backend="cv2", grayscale=True) / 255.0
        assert label.ndim == 2

        if image.shape[:2] != label.shape:
            print("image and label shape mismatch, reshape.")
            H, W = image.shape[:2]
            label = cv2.resize(label, (W, H))

        flip = self.flip and np.random.choice([True, False])
        if flip:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)

        if np.all(self.imsize > 0):
            H, W = self.imsize
            image = cv2.resize(image, (W, H))
            label = cv2.resize(label, (W, H))

        image = image.transpose((2,0,1))
        label = label[np.newaxis, :, :]
        
        metas = dict({
            "filename": img_fullname,
            "flip": flip
        })
        result = dict({
            "image": image.astype(np.float32),
            "label": label.astype(np.float32),
            "metas": metas
        })

        return result

    def __len__(self):
        return len(self.__item_names)

    def get_item_names(self):
        return self.__item_names.copy()