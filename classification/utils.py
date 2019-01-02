import shutil, sys, os, torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import yaml
from os.path import join, split, abspath, dirname, isfile, isdir

def load_yaml(yaml_file):
    assert isfile(yaml_file), "File %s does'nt exist!" % yaml_file
    return yaml.load(open(yaml_file))

def merge_config(args, yaml_config):

    if hasattr(args, "data") and args.data != yaml_config["DATA"]["DIR"]:
        yaml_config["DATA"]["DIR"] = args.data

    if hasattr(args, "tmp") and args.tmp != yaml_config["MISC"]["TMP"]:
        yaml_config["MISC"]["TMP"] = args.tmp

    # Optimizer
    if hasattr(args, "bs") and args.data != yaml_config["OPTIMIZER"]["BS"]:
        yaml_config["OPTIMIZER"]["BS"] = args.bs

    if hasattr(args, "lr") and args.lr != yaml_config["OPTIMIZER"]["LR"]:
        yaml_config["OPTIMIZER"]["LR"] = args.lr
    
    # CUDA
    if hasattr(args, "gpu") and args.gpu != yaml_config["CUDA"]["GPU_ID"]:
        yaml_config["CUDA"]["GPU_ID"] = args.gpu
    
    if hasattr(args, "visport") and args.gpu != yaml_config["VISDOM"]["PORT"]:
        yaml_config["VISDOM"]["PORT"] = args.visport

    return yaml_config