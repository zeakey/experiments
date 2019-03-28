import torch

def MAE(prediction, target, expectation=True):
    assert prediction.squeeze().ndimension() == 2, prediction.squeeze().shape
    assert target.squeeze().ndimension() == 1
    if expectation:
        N, K = prediction.shape
        values = torch.arange(K).float().cuda().view(1, -1).expand_as(prediction)
        result = (prediction * values).sum(dim=1)
        return torch.abs(target.float() - result).mean().item()
    else:
        return torch.abs(target - torch.argmax(prediction, dim=1)).float().mean().item()
