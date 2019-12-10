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


def verification(features, pair_label, nfolds=10):
    """
    features: image features, shape: [2*N, D]
    pair_label: pairwise labels, shape [N]

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
    assert isinstance(pair_label, np.ndarray)
    assert features.shape[0] == pair_label.size*2

    mu = features.mean(axis=0).reshape(1, -1)
    std = features.std(axis=0).reshape(1, -1)
    features -= mu
    features /= std
    # compute pairwise distance
    # L2 normalize features
    features = sklearn.preprocessing.normalize(features)
    f1 = features[0::2]
    f2 = features[1::2]
    # cosine distance
    distance = -(f1*f2).sum(axis=1)
    assert distance.max() <= 1 and distance.min() >= -1, "maximal distance: %f, minimal distance: %f"%(distance.max(), distance.min())

    folds = KFold(n_splits=nfolds, shuffle=False)
    thresholds = np.linspace(-10, 10, 1000)

    accuracy = np.zeros((nfolds,))

    for i, (val_idx, test_idx) in enumerate(folds.split(np.arange(distance.shape[0]))):

        test_distance = distance[test_idx]
        test_label = pair_label[test_idx]

        val_distance = distance[val_idx]
        val_label = pair_label[val_idx]

        acc = search_threshold(val_distance, val_label, thresholds)
        optimal_thres = thresholds[np.argmax(acc)]

        pred = test_distance <= optimal_thres
        accuracy[i] = (pred == test_label).mean()
    return accuracy

def search_threshold(distance, pair_label, thresholds):
    assert distance.size == pair_label.size
    acc = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        pred = distance <= t
        acc[i] = (pred == pair_label).mean()
    return acc

if __name__ == "__main__":

    features = np.load("lfw-features.npy")
    pair_label = np.load("lfw-labels.npy")
    acc= verification(features, pair_label, nfolds=10)
    print(acc.mean())


# REFERENCE
# [1] FaceNet: A Unified Embedding for Face Recognition and Clustering