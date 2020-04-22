import numpy as np
import pydensecrf.densecrf as dcrf

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

    crf_prediction = np.zeros_like(maps)
    for i in range(N):
        S = np.squeeze(state[i])
        I = np.squeeze(images[i])

        crfed = crf_single_image(I, S)
        crf_prediction[i, ] = crfed

    return crf_prediction


    
