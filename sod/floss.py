import torch
import torch.nn as nn

class FLoss(nn.Module):
    def __init__(self, beta=0.3, log_like=False):
        super(FLoss, self).__init__()
        self.beta = beta
        self.log_like = log_like

    def forward(self, prediction, target):
        EPS = 1e-5
        N = prediction.size(0)
        prediction = torch.sigmoid(prediction)
        TP = (prediction * target).view(N, -1).sum(dim=1)
        H = self.beta * target.view(N, -1).sum(dim=1) + prediction.view(N, -1).sum(dim=1)
        fmeasure = (1 + self.beta) * TP / (H + EPS)
        if self.log_like:
            floss = -torch.log(fmeasure)
        else:
            floss  = (1 - fmeasure)
        return floss.mean()

# copied from deep-usps
def F_cont(sal_pred, Disc_Label, b=1.5):
    assert sal_pred.shape==Disc_Label.shape
    # Get True Positives, False Positives, True Negatives (Continuous!)
    TP=sal_pred*Disc_Label
    FP=sal_pred*(1-Disc_Label)
    TN=(1-sal_pred)*Disc_Label
    # sum up TP,FP, for each image
    if int(torch.__version__[0])>0:
        TP=torch.sum(TP, dim=(1,2))
        FP=torch.sum(FP, dim=(1,2))
        TN=torch.sum(TN, dim=(1,2))
    else: # the above does not work in torch 0.4, which we need for ffi for Deeplab
        TP=torch.sum(torch.sum(TP, dim=2), dim=1)
        FP=torch.sum(torch.sum(FP, dim=2), dim=1)
        TN=torch.sum(torch.sum(TN, dim=2), dim=1)
    eps=1e-5
    prec=TP/(TP+FP+eps)
    recall=TP/(TP+TN+eps)
 
    F=(1+b)*prec*recall/(b*prec+recall+eps)
    Loss=1-F

    return torch.mean(Loss)