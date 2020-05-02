import argparse, os
import numpy as np
from PIL import Image
from os.path import join, isdir
import matplotlib.pyplot as plt

EPSILON = 1e-6

parser = argparse.ArgumentParser(description='Evaluate saliency detection results')
parser.add_argument('--pred', required=True, type=str)
parser.add_argument('--gt', required=True)
parser.add_argument('--H', type=int, default=432)
parser.add_argument('--W', type=int, default=432)
parser.add_argument('--fig', type=str, default="prcurve.pdf")
parser.add_argument('--num_thres', type=int, default=50)



def prf(pred_dir, gt_dir, num_thres=50):
    # gt_names = [i for i in os.listdir(args.gt) if i.endswith(".png")]
    pred_names = [i for i in os.listdir(pred_dir) if i.endswith(".png")]
    N = len(pred_names)

    # assert len(gt_names) == len(pred_names)

    N = len(pred_names)

    gt = np.zeros((N, args.H, args.W), dtype=bool)
    pred = np.zeros((N, args.H, args.W), dtype=np.uint8)

    for idx, name in enumerate(pred_names):
        g = Image.open(join(args.gt, name)).convert("L").resize((args.W, args.H), Image.NEAREST)
        p = Image.open(join(pred_dir, name)).convert("L").resize((args.W, args.H), Image.BILINEAR)

        g = np.array(g)
        p = np.array(p)

        assert np.unique(g).size == 2

        gt[idx] = g != 0
        pred[idx] = p


    precision = np.zeros((args.num_thres,), dtype=np.float32)
    recall = np.zeros((args.num_thres,), dtype=np.float32)


    print("Evaluating %d salient maps from %s" % (pred_dir, N))
    for idx, t in enumerate(np.linspace(0, 255, args.num_thres).tolist()):
        p = pred >= t

        if np.count_nonzero(p) == 0:
            precision[idx] = 1
            recall[idx] = 0
            break

        TP = np.count_nonzero(p * gt)
        FP = np.count_nonzero(p * (1 - gt))
        FN = np.count_nonzero((1 - p) * gt)

        p = TP / max(TP + FP, EPSILON)
        r = TP / max(TP + FN, EPSILON)

        precision[idx] = p
        recall[idx] = r

        print("threshold %d precision %.4f recall %.4f" % (t, p, r))

    precision[precision < EPSILON] = EPSILON
    recall[recall < EPSILON] = EPSILON

    return precision, recall

if __name__ == "__main__":
    args = parser.parse_args()
    assert isdir(args.gt)
    pred_dirs = args.pred.split(',')

    precision = np.zeros((len(pred_dirs), args.num_thres), dtype=np.float32)
    recall = np.zeros((len(pred_dirs), args.num_thres), dtype=np.float32)

    for pred_dir in pred_dirs:
        assert isdir(pred_dir), pred_dir

        p, r = prf(pred_dir, args.gt, args.num_thres)
        plt.plot(p, r)
    plt.grid()
    plt.legend(pred_dirs)
    plt.xlabel("precision")
    plt.ylabel("recall")
    plt.savefig(args.fig)
