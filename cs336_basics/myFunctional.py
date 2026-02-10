import os
import torch
import torch.nn as nn

def toy_softmax(x):
    sx = x -  x.max(-1,keepdim=True).values
    ex = sx.exp()
    return ex/ex.sum(-1,keepdim=True)