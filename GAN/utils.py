import torch

def analyse_spectral(model):
    spectral = dict()
    for name, p in model.named_parameters():
        if "weight" in name and len(p.shape) >= 2:
            _, sv, _ = torch.svd(p.data.view(p.shape[0], -1))
            spectral[name] = sv
    return spectral
