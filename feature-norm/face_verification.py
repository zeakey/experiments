import numpy as np
import sklearn
from sklearn.model_selection import KFold

def val_far(pred, label):
    """
    calculate validation rate (VAL) and false accept rate (FAR)
    See definitions in Eq.6 of [1]
    """
    true_positive = np.logical_and(pred, label).sum()
    false_positive = np.logical_and(pred, np.logical_not(label)).sum()
    val = true_positive / label.sum()
    far = false_positive / (1-label).sum()
    return val, far


def verification(features, labels, nfolds=10):
    """
    features: image features, shape: [2*N, D]
    labels: pairwise labels, shape [N]

    the features should be organised as:
    [
    pair1_f1,
    pair1_f2,
    pair2_f1,
    pair2_f2,
    ...
    ]
    """
    assert isinstance(features, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert features.shape[0] == labels.size*2

    folds = KFold(n_splits=nfolds, shuffle=False)
    thresholds = np.linspace(-1, 1, 1000)

    accuracy = np.zeros((nfolds,))

    for i, (val_idx, test_idx) in enumerate(folds.split(np.arange(labels.size))):

        features1 = features.copy()
        # whiten featuress
        f1 = features1[0::2]
        f2 = features1[1::2]
        valfeat = np.concatenate((f1[val_idx,], f2[val_idx,]), axis=0)

        mu = valfeat.mean(axis=0)
        std = valfeat.std(axis=0)
        features1 -= mu
        # features1 /= std

        # L2 normalize features
        features1 = sklearn.preprocessing.normalize(features1)

        # cosine distance
        f1 = features1[0::2]
        f2 = features1[1::2]
        distance = -(f1*f2).sum(axis=1)
        assert distance.max() <= 1 and distance.min() >= -1, "maximal distance: %f, minimal distance: %f"%(distance.max(), distance.min())

        test_distance = distance[test_idx]
        test_label = labels[test_idx]

        val_distance = distance[val_idx]
        val_label = labels[val_idx]

        acc = search_threshold(val_distance, val_label, thresholds)
        optimal_thres = thresholds[np.argmax(acc)]

        pred = test_distance <= optimal_thres
        accuracy[i] = (pred == test_label).mean()
    return accuracy

def search_threshold(distance, label, thresholds):
    assert distance.size == label.size
    acc = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        pred = distance <= t
        acc[i] = (pred == label).mean()
    return acc

if __name__ == "__main__":

    import torch
    from models import sphere_face

    model = sphere_face.Sphere20()
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    # model.load_state_dict(torch.load("tmp/sphereface-norm-grad-r4/model_best.pth")["state_dict"])

    data = torch.load("lfw-112x112.pth", map_location=torch.device('cpu'))
    label = data["label"].numpy()
    data = data["data"].float()
    data = data - 127.5
    data = data * 0.0078125

    # print(data.mean(dim=(0,2,3)), data.std(dim=(0,2,3)))
    features = np.zeros((12000, 512), dtype=np.float32)
    for i in range(0, 12000, 100):
        d1 = data[i:i+100,].cuda()
        d2 = torch.flip(d1, dims=(3,))
        o1 = model(d1)
        o2 = model(d2)

        if True:
            o = o1+o2
        else:
            o = torch.cat((o1, o2), dim=1)
        o = o.detach().cpu().numpy()

        features[i:i+100,] = o
    acc= verification(features, label, nfolds=10)
    print(acc.mean())

    import test_lfw
    test_lfw.fold10(features, label)

# REFERENCE
# [1] FaceNet: A Unified Embedding for Face Recognition and Clustering