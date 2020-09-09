import torch
from PIL import Image
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from os.path import join, isdir, isfile
from skimage.filters import sobel_h, sobel_v
import cv2

def seg2edge(seg):
    grad = np.gradient(seg.astype(np.float32))
    grad = np.sqrt(grad[0]**2 + grad[1]**2)

    return grad != 0

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

def dist2flux(dist):
    sobelx = cv2.Sobel(dist,cv2.CV_64F,1,0,ksize=9)
    sobely = cv2.Sobel(dist,cv2.CV_64F,0,1,ksize=9)

    flux = np.dstack((sobely, sobelx)).transpose((2,0,1))

    # normalize
    norm = np.expand_dims(np.sqrt((flux**2).sum(axis=0)), axis=0)
    norm[norm==0] = 1
    flux /= norm

    return flux

def flux2angle(flux):
    """
    flux: a [2, H, W] tesnro represents the flux vector of each position
    """
    _, H, W = flux.shape
    top_half = flux[0,...] >= 0 # y >= 0, \theta <= \pi
    bottom_half = flux[0,...] < 0 # y < 0, \theta > \pi
    
    unit = np.zeros((2, H, W), dtype=np.float32)
    unit[1,...] = 1 # unit vector: (y=0, x=1)
    cos = (flux * unit).sum(axis=0)
    acos = np.arccos(cos)
    angle = np.zeros((H, W), dtype=np.float32)
    angle[top_half] = acos[top_half]
    angle[bottom_half] = 2*np.pi - acos[bottom_half]

    return angle

class SODDatasetDual(torch.utils.data.Dataset):
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
        self.__item_names = [line.strip() for line in open(name_list, 'r')]

    def __getitem__(self, index):
        assert isfile(join(self.img_dir, self.__item_names[index]+".jpg"))
        img = Image.open(join(self.img_dir, self.__item_names[index]+".jpg"))

        labels = []
        for label_dir in self.label_dirs:
            assert isfile(join(label_dir, self.__item_names[index]+".png"))
            labels.append(Image.open(join(label_dir, self.__item_names[index]+".png")))

        if self.flip and np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            labels = [label.transpose(Image.FLIP_LEFT_RIGHT) for label in labels]

        if self.image_transform:
            img = self.image_transform(img)
        if self.label_transform:
            labels = [self.label_transform(l) for l in labels]

        return tuple([img]+labels+[index])

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
        return len(self.__item_names)

    def get_item_names(self):
        return self.__item_names.copy()


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
        assert isfile(join(self.img_dir, self.__item_names[index]+".jpg"))
        img_fullname = join(self.img_dir, self.__item_names[index]+".jpg")
        image = Image.open(img_fullname)

        label = Image.open(join(self.label_dir, self.__item_names[index]+".png"))

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

        angle = flux2angle(dist2flux(dist))
        # clamp angle
        angle[angle>=np.pi*2] = np.pi*2 - 1e-5
        orientation = np.floor(angle / (np.pi*2/8)).astype(np.uint8)

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
