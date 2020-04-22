import numpy as np
from PIL import Image
from os.path import join, isfile
import warnings, os

def load_batch_images(names, H=432, W=432):
    assert isinstance(names, list)

    images = np.zeros((len(names), 3, H, W), dtype=np.uint8)
    for idx, n in enumerate(names):
        assert isfile(n), n
        im = Image.open(n).convert("RGB").resize((W, H))
        im = np.array(im, dtype=np.uint8)
        images[idx, :, :, :] = im.transpose((2, 0, 1))
    return images

def get_full_image_names(dir, names, ext=".jpg", warn_exist=False):
    assert isinstance(names, list)
    assert ext.startswith("."), "%s is not a valid extension" % ext
    full_names = []
    for n in names:
        fulln = join(dir, n+ext)
        full_names.append(fulln)
        if warn_exist and not isfile(fulln):
            warnings.warn("%s not exist!"%fulln)
    return full_names

def save_maps(maps, names, dir, ext=".png"):
    assert isinstance(maps, np.ndarray)
    assert isinstance(names, list)
    assert maps.dtype == np.uint8 and maps.ndim == 4
    assert ext.startswith("."), "%s is not a valid extension" % ext
    assert maps.shape[1] == 1 or maps.shape[1] == 3, "maps shape: {}".format(maps.shape)
    os.makedirs(dir, exist_ok=True)
    N = maps.shape[0]

    for idx in range(N):
        p = np.squeeze(maps[idx].transpose((1,2,0)))
        p = Image.fromarray(p)
        p.save(join(dir, names[idx]+ext))