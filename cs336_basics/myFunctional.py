import torch
import torch.nn as nn
import os

def toy_softmax(x):
    sx = x -  x.max(-1,keepdim=True).values
    ex = sx.exp()
    return ex/ex.sum(-1,keepdim=True)

def toy_product_atte(Q, K, V, mask=None):
    d_k = torch.tensor(Q.shape[-1])
    Qk = Q @ K.transpose(-2,-1)/ torch.sqrt(d_k)
    if mask is not None:
        Qk = Qk.masked_fill(mask==False,float('-inf'))
    sQk = toy_softmax(Qk)
    return sQk @ V

def toy_cross_entry(logits, targets):
    #loss = -exp(logits)/sum(exp(logits))[pi].log().mean()
    #     = -(logits[pi] - sum(exp(logits).log)).mean()
    m_logits = logits - logits.max(-1,keepdim=True).values
    log_probs =m_logits - m_logits.exp().sum(-1,keepdim = True).log()
    t = torch.arange(len(targets))
    return -(log_probs[t,targets]).mean()