import os
import shutil
from os.path import join, isdir

origin = "/media/data2/dataset/ilsvrc12"
dest = "/media/data0/ilsvrc12-subset"

os.makedirs(join(dest, "train"), exist_ok=True)
os.makedirs(join(dest, "val"), exist_ok=True)

train_dir = join(origin, "train")
val_dir = join(origin, "val")

categories = [c for c in os.listdir(train_dir) if isdir(join(train_dir, c))]

assert len(categories) == len([c for c in os.listdir(val_dir) if isdir(join(val_dir, c))])

for c in [c for c in os.listdir(val_dir) if isdir(join(val_dir, c))]:
    assert c in categories


categories = sorted(categories)

for i in range(0, len(categories), 5):
    c = categories[i]
    frm = join(origin, "train", c)
    to  = join(dest, "train", c)
    shutil.copytree(frm, to)

    frm = join(origin, "val", c)
    to  = join(dest, "val", c)
    shutil.copytree(frm, to)

    print("Category %d of %d" % (i, len(categories)))
