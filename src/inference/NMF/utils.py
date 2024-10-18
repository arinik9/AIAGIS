
import torch

def BCELoss(pred_y, true_y):
    cross_loss=torch.mean(-true_y*torch.log(torch.clip(pred_y,1e-10,1.0))-(1.0-true_y)*torch.log(torch.clip((1.0-pred_y),1e-10,1.0)))
    return cross_loss

def MAE(pred_y, true_y):
    return torch.mean(torch.abs(pred_y-true_y)).item()