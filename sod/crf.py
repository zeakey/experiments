import torch
import numpy as np
import pydensecrf.densecrf as dcrf
from torch.utils.data.sampler import BatchSampler, SequentialSampler
import time, sys
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import multiprocessing

def crf_single_image(image, state):
    """
    state: [2, H, W] tensor
    image: a [H, W, C] image
    """
    assert isinstance(state, np.ndarray)
    assert isinstance(image, np.ndarray)
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
        d.setUnaryEnergy(-np.log(state.reshape(2, H*W).astype(np.float32)))

    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=image, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    new_state = d.inference(5)
    new_state = np.array(new_state).reshape((2, H, W))

    return np.squeeze(new_state[1, :, :])

def batch_crf(images, maps):
    """
    apply densecrf to a batch of maps
    images, maps: N, C, H, W format tensors
    """
    assert isinstance(images, np.ndarray) and isinstance(maps, np.ndarray)
    assert maps.min() <= 1 and maps.max() >= 0, "maps.min() %f v.s. maps.max() %f" % (maps.min(), maps.max())
    assert images.ndim == 4 and maps.ndim == 4
    N, _, H, W = images.shape
    assert maps.shape == (N, 1, H, W)

    assert maps.ndim == 4
    state = np.concatenate((1-maps, maps), axis=1)
    assert state.ndim == 4 and state.shape[1] == 2
    state = np.ascontiguousarray(state)

    images = images.transpose((0, 2, 3, 1))
    images = np.ascontiguousarray(images)

    crfed_maps = np.zeros_like(maps)
    for i in range(N):
        S = np.squeeze(state[i])
        I = np.squeeze(images[i])

        crfed = crf_single_image(I, S)
        crfed_maps[i, ] = crfed

    return crfed_maps

def par_batch_crf(images, maps, num_works=12):
    """
    multiprocessing parallel batch crf
    """
    assert isinstance(images, np.ndarray) and isinstance(maps, np.ndarray)
    assert maps.min() <= 1 and maps.max() >= 0, "maps.min() %f v.s. maps.max() %f" % (maps.min(), maps.max())
    assert images.ndim == 4 and maps.ndim == 4
    N, _, H, W = images.shape
    assert maps.shape == (N, 1, H, W), maps.shape

    assert maps.ndim == 4
    state = np.concatenate((1-maps, maps), axis=1)
    assert state.ndim == 4 and state.shape[1] == 2
    state = np.ascontiguousarray(state)

    images = images.transpose((0, 2, 3, 1))
    images = np.ascontiguousarray(images)

    crfed_maps = np.zeros_like(maps)

    def crf_thread(images, states, crfed_maps, work_length, start_pos):
        for i in range(work_length):
            S = np.squeeze(state[i])
            I = np.squeeze(images[i])
            t = time.time()
            crfed = crf_single_image(I, S)
            crfed_maps[i + start_pos, ] = crfed

    offset = N // num_works
    remain = N % num_works
    threads = []
    for i in range(num_works):
        if i == num_works-1:
            length = offset + remain
        else:
            length = offset
        params = [images, state, crfed_maps, length, i*offset]
        threads.append(multiprocessing.Process(target=crf_thread, args=(params)))
    for i in range(num_works):
        threads[i].start()
    for i in range(num_works):
        threads[i].join()

    return crfed_maps


class CRFDataset(torch.utils.data.Dataset):
    def __init__(self, maps, images):
        assert images.dtype == np.uint8, images.dtype
        assert maps.shape[0] == images.shape[0]
        assert maps.max() <= 1 and maps.min() >= 0

        self.maps = maps
        # [N C H W] -> [N H W C]
        self.images = np.ascontiguousarray(images.transpose((0, 2, 3, 1)))

    def __getitem__(self, index):
        im = self.images[index]
        state = np.squeeze(self.maps[index])
        state = np.stack((1-state, state), axis=0)
        crf = crf_single_image(im, state)

        return crf, index

    def __len__(self):
        return self.images.shape[0]

def par_batch_crf_dataloader(images, maps):
    """
    Using pytorch's parallel dataloader for parallel crf
    """
    crf_dataset = CRFDataset(maps, images=images)
    batch_sampler = BatchSampler(SequentialSampler(crf_dataset), batch_size=10, drop_last=False)
    crf_loader = torch.utils.data.DataLoader(
         crf_dataset,
         batch_sampler=batch_sampler,
         shuffle=False,
         num_workers= min(images.shape[0], 12),
         pin_memory=True,
         drop_last=False)
    
    crfed_maps = np.zeros_like(maps)
    for crf, indx in crf_loader:
        crfed_maps[indx, 0] = crf

    return crfed_maps

if __name__ == "__main__":
    images = (np.random.rand(100, 3, 432, 432)*255).astype(np.uint8)
    maps = np.random.rand(100, 1, 432, 432)
    start = time.time()
    par_batch_crf_dataloader(images, maps)
    print("par_batch_crf_dataloader: %f sec"%(time.time()-start))

    start = time.time()
    par_batch_crf(images, maps)
    print("par_batch_crf: %f sec"%(time.time()-start))

    start = time.time()
    batch_crf(images, maps)
    print("batch_crf: %f sec"%(time.time()-start))
