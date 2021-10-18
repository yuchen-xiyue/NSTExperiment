import torch
import torch.nn as nn


class GramMatrix(nn.Module):
    def forward(self, X):
        b, c, h, w = X.size()
        X = X.view(b * c, h * w)
        return torch.mm(X, X.T).div_(h * w)


class GramMSELoss(nn.Module):
    def forward(self, X, Y):
        return nn.MSELoss()(GramMatrix()(X), Y)


class ContentLoss(nn.Module):
    def forward(self, X, Y):
        return nn.MSELoss()(X, Y)
