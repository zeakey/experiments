import torch
import numpy as np
from PIL import Image
import pydensecrf.densecrf as dcrf
from torch.utils.data.sampler import BatchSampler, SequentialSampler
import time, sys
from os.path import isfile
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import multiprocessing
from skimage.io import imsave
import os, sys

def crf_single_image(image, state):
    """
    state: [2, H, W] tensor
    image: a [H, W, C] image
    """
    assert isinstance(state, np.ndarray)
    assert isinstance(image, np.ndarray), type(image)
    assert state.dtype == np.float32
    assert state.ndim == 3, "state shape {}".format(state.shape)
    assert state.max() <= 1 and state.min() >= 0, "state.min()=%f, state.max()=%f"%(state.min(), state.max())

    _ , H, W = state.shape
    assert image.shape[0]==H and image.shape[1]==W

    # in case of gray image
    if image.ndim == 2:
        image=np.stack((image, image, image), axis=2)
    assert image.ndim == 3

    d = dcrf.DenseCRF2D(W, H, 2)

    with np.errstate(divide='ignore', invalid='ignore'):
        d.setUnaryEnergy(-np.log(state.reshape(2, H*W)))

    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=image, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    new_state = d.inference(5)
    new_state = np.array(new_state).reshape((2, H, W))

    return np.squeeze(new_state[1, :, :])

class CRFDataset(torch.utils.data.Dataset):
    def __init__(self, images, maps, save_dir, H=432, W=432):
        self.maps = maps
        self.images = images
        self.save_dir = save_dir
        self.H, self.W = H, W

    def __getitem__(self, index):
        im = np.array(Image.open(self.images[index]).resize((self.W, self.H)))
        m = np.array(Image.open(self.maps[index]).resize((self.W, self.H)))
        assert m.ndim == 2
        state = m.astype(np.float32) / 255.0
        state = np.stack((1-state, state), axis=0)
        crf = crf_single_image(im, state)
        crf = (crf * 255).astype(np.uint8)

        fn = os.path.splitext(os.path.split(self.images[index])[-1])[0]
        fp = os.path.join(self.save_dir, fn+".png")
        Image.fromarray(crf).save(fp)

        # for visualization
        m = np.stack((m,m,m), axis=2)
        crf = np.stack((crf,crf,crf), axis=2)
        vis = np.concatenate((im, m, crf), axis=1)
        Image.fromarray(vis).save(os.path.join(self.save_dir, fn+"-im-prediction-crf.jpg"))

        return fp, index

    def __len__(self):
        return len(self.images)

def par_batch_crf_dataloader(images, maps, save_dir, num_workers=12):
    """
    Using pytorch's parallel dataloader for parallel crf
    """
    assert type(images) == list
    assert type(maps) == list
    for i in images:
        assert isfile(i)
    for i in maps:
        assert isfile(i)
    os.makedirs(save_dir, exist_ok=True)
    crf_dataset = CRFDataset(images, maps, save_dir)
    batch_sampler = BatchSampler(SequentialSampler(crf_dataset), batch_size=10, drop_last=False)
    crf_loader = torch.utils.data.DataLoader(
         crf_dataset,
         batch_sampler=batch_sampler,
         shuffle=False,
         num_workers= min(len(images), num_workers),
         pin_memory=True,
         drop_last=False)

    crf_names = []
    for fp, idx in crf_loader:
        crf_names += fp

    return crf_names



if __name__ == "__main__":
    images = (np.random.rand(100, 3, 432, 432)*255).astype(np.uint8)
    maps = (np.random.rand(100, 1, 432, 432) * 255).astype(np.uint8)
    start = time.time()
    par_batch_crf_dataloader(images, maps)
    print("par_batch_crf_dataloader: %f sec"%(time.time()-start))

    start = time.time()
    par_batch_crf(images, maps)
    print("par_batch_crf: %f sec"%(time.time()-start))

    start = time.time()
    batch_crf(images, maps)
    print("batch_crf: %f sec"%(time.time()-start))
