import numpy as np
from PIL import Image
from os.path import join, isfile
import warnings, os

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name="value", fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def load_batch_images(names, H=432, W=432):
    assert isinstance(names, list)

    images = np.zeros((len(names), 3, H, W), dtype=np.uint8)
    for idx, n in enumerate(names):
        assert isfile(n), n
        im = Image.open(n).resize((W, H))
        if im.mode == "RGB":
            im = np.array(im, dtype=np.uint8)
            images[idx, :, :, :] = im.transpose((2, 0, 1))
        else:
            images[idx, 0, :, :] = np.array(im, dtype=np.uint8)

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
    fp = []

    for idx in range(N):
        p = np.squeeze(maps[idx].transpose((1,2,0)))
        p = Image.fromarray(p)
        p.save(join(dir, names[idx]+ext))
        fp.append(join(dir, names[idx]+ext))
    return fp

def mva_single_image(old, new, alpha=0.7):
    assert isinstance(old, np.ndarray)
    assert isinstance(new, np.ndarray)
    assert old.dtype == np.uint8
    assert new.dtype == np.uint8

    old = old.astype(np.float32) / 255
    new = new.astype(np.float32) / 255

    mva = old*alpha + new*(1-alpha)
    mva = (mva*255).astype(np.uint8)

    return mva

def batch_mva(old, new, save_dir, alpha=0.7):
    assert isinstance(old, list)
    assert isinstance(new, list)
    assert len(old) == len(new)
    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(old)):
        assert isfile(old[i]), old[i]
        assert isfile(new[i]), new[i]

        im_old = np.array(Image.open(old[i]))
        im_new = np.array(Image.open(new[i]))

        mva = mva_single_image(im_old, im_new, alpha)

        fp = join(save_dir, os.path.split(old[i])[-1])
        Image.fromarray(mva).save(fp)

def f_zero_point_five(pred, gt, beta=0.3):
    """
    F-measure at threshold=0.5
    """
    assert pred.max() <= 1 and pred.min() >= 0
    EPS = 1e-6

    pred = pred >= 0.5
    TP = (pred * gt).sum()
    H = beta * gt.sum() + pred.sum()

    F = (1 + beta) * TP/ (H + EPS)

    return F.mean()


def evaluate_maps(prediction_dir, gt_dir, names, size=(432, 432), ext=".png"):
    mae = AverageMeter()
    fmeasure = AverageMeter()
    for i in names:
        assert isfile(join(prediction_dir, i+ext)), join(prediction_dir, i+ext)
        assert isfile(join(gt_dir, i+ext)), join(gt_dir, i+ext)

        pred = np.array(Image.open(join(prediction_dir, i+ext)).resize(size)) / 255.0
        gt = np.array(Image.open(join(gt_dir, i+ext)).resize(size)) / 255.0

        mae.update(np.abs(gt-pred).mean())
        fmeasure.update(f_zero_point_five(pred, gt))
    return mae.avg, fmeasure.avg



