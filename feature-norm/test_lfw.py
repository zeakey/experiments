# Code written by KAIZ
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from sklearn.decomposition import PCA
from os.path import isfile, isdir, join, split, splitext
import datetime, argparse
EPSILON = np.float32(1e-9)

def l2norm(x):
  """
  L2 norm array along the 1-th axis
  """
  l2 = np.linalg.norm(x, axis=1)
  l2[l2 == 0] = EPSILON
  x = x.transpose() / l2
  return x.transpose()

def tune_thres(similarity, label, nthres=20000):
  """
  tune the best threshold
  """
  if similarity.size != label.size:
    print(similarity.shape, label.shape)
  assert similarity.size == label.size, "%d vs %d" % (similarity.size, label.size)
  # assert similarity.max() <= 1,  similarity.max()
  # assert similarity.min() >= -1, similarity.min()
  assert np.unique(label).size <= 2
  best_acc, best_thres = np.float32(-1), np.float32(-1)
  for t in np.linspace(-1, 1, nthres):
    predict = (similarity >= t)
    acc = np.mean(np.float32(predict == label))
    if acc >= best_acc:
      best_acc, best_thres = acc, t
  return best_acc, best_thres

def get_similarity(features):
  N = features.shape[0]
  assert N % 2 == 0
  features = l2norm(features)
  features0 = features[np.arange(0, N, 2), :]
  features1 = features[np.arange(0, N, 2) + 1, :]
  return np.sum(np.multiply(features0, features1), 1)

def fold10(features, labels, cache_fn='lfw_result.txt', silent=False):
  if not silent:
    print("=========start 10-fold cross-validation=========")
  N, K = features.shape
  cache_data = np.zeros((11, 2), np.float32)
  for i in range(10):
    # indice of validation and testing pairs
    test_pair_idx = list(range(600*i, 600*(i+1)))
    test_img_idx  = list(range(1200*i, 1200*(i+1)))
    #test_pair_idx = list(range(i * (N // 40), (i + 1) * (N // 40))) + list(range(i * (N // 40) + N // 4, (i + 1) * (N // 40) + N // 4))
    #test_img_idx  = list(range(i * (N // 20), (i + 1) * (N // 20))) + list(range(i * (N // 20) + N // 2, (i + 1) * (N // 20) + N // 2))
    val_pair_idx = [idx for idx in range(N // 2) if idx not in test_pair_idx]
    val_img_idx = [idx for idx in range(N) if idx not in test_img_idx]
    num_test_pair, num_val_pair = len(test_pair_idx), len(val_pair_idx)
    num_test_img, num_val_img = len(test_img_idx), len(val_img_idx)
    # testing features and validation features
    # get similarity
    mu = np.mean(features[val_img_idx], 0)
    similarity = get_similarity(features - mu)
    val_similarity = similarity[val_pair_idx]
    # labels of validation set
    val_label = labels[val_pair_idx]

    # tune the best thres
    best_acc, best_thres = tune_thres(val_similarity, val_label)
    # test using the best threshold
    test_label = labels[test_pair_idx]
    test_similarity = similarity[test_pair_idx]
    predict = test_similarity >= best_thres
    test_acc = np.mean(np.float32(predict == test_label))
    cache_data[i, :] = np.array([best_thres, test_acc])
    if not silent:
      print("fold %d: accuracy = %f, best threshold = %f" % (i+1, test_acc, best_thres))
  if not silent:
    print("Average accuracy: %f, average threshold: %f " % (cache_data[0:10, 1].mean(), cache_data[0:10, 0].mean()))
    print("=========The End=========")
  cache_data[10, :] = np.array([best_thres, cache_data[0:10, 1].mean()])
  np.savetxt(cache_fn, cache_data, fmt="%.5f")
  if not silent:
    print("Done, evaluation results have been saved at \"%s\"" % cache_fn)
  return cache_data[0:10, 1].mean()

