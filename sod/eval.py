import os, sys
from os.path import join, splitext, isdir, abspath
import multiprocessing

# sal metric executive
exe = "/home/kai/Code0/others/PoolNet/SalMetric/build/salmetric"

if len(sys.argv) != 4:
    print("Usage: python eval.py prediction_dir gt_dir log")
    sys.exit()

assert isdir(sys.argv[1]) and isdir(sys.argv[2])

items = [i for i in os.listdir(sys.argv[2]) if i.endswith(".png")]

f = open("tmp.txt", "w")
for i in items:
    pred_fn = splitext(i)[0]+"_sal_fuse.png"
    pred_fn = abspath(join(sys.argv[1], pred_fn))

    gt_fn = abspath(join(sys.argv[2], i))

    line = pred_fn + " " + gt_fn + "\n"
    f.write(line)
f.close()

nthres = multiprocessing.cpu_count()
cmd = "%s tmp.txt %d 2>&1 | tee %s" % (exe, nthres, sys.argv[3])

print("Start to evaluate with command '%s'..."%cmd)
os.system(cmd)