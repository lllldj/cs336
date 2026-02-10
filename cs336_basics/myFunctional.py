import os
import torch
import torch.nn as nn
from cs336_basics.myModule import toy_RoPE


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


#never used in multi_head attention.
def toy_multihead_atte(d_model,num_heads,Qp,Kp,Vp,proj,in_features,posistion=None):
    Qs = (in_features @ Qp.transpose(-2,-1)).split(d_model//num_heads,-1)
    Ks = (in_features @ Kp.transpose(-2,-1)).split(d_model//num_heads,-1)
    Vs = (in_features @ Vp.transpose(-2,-1)).split(d_model//num_heads,-1)
    
    seq_len = Qs[0].size(-2)
    mask = torch.tril(torch.ones(seq_len,seq_len))
    
    atts = [toy_product_atte(Qs[i],Ks[i],Vs[i],mask) for i in range(num_heads)]
    atts = torch.cat(atts,-1)
    return atts @  proj.transpose(-2,-1)

#never used in multi_head attention.
def toy_multihead_atte_rope(d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    Qp, #Float[Tensor, " d_k d_in"],
    Kp, #Float[Tensor, " d_k d_in"],
    Vp, #Float[Tensor, " d_v d_in"],
    proj, #Float[Tensor, " d_model d_v"],
    in_features, #Float[Tensor, " ... sequence_length d_in"],
    token_positions, #Int[Tensor, " ... sequence_length"] | None = None,
):# -> Float[Tensor, " ... sequence_length d_out"]
    Ro = toy_RoPE(d_model//num_heads,theta,max_seq_len) #
    Qs = (in_features @ Qp.transpose(-2,-1)).split(d_model//num_heads,-1)
    Ks = (in_features @ Kp.transpose(-2,-1)).split(d_model//num_heads,-1)
    Vs = (in_features @ Vp.transpose(-2,-1)).split(d_model//num_heads,-1)
    
    Qs = [Ro.forward(Qs[i],token_positions) for i in range(num_heads)] #
    Ks = [Ro.forward(Ks[i],token_positions) for i in range(num_heads)]
    
    seq_len = Qs[0].size(-2)
    mask = torch.tril(torch.ones(seq_len,seq_len))
    atts = [toy_product_atte(Qs[i],Ks[i],Vs[i],mask) for i in range(num_heads)]
    atts = torch.cat(atts,-1)
    return atts @  proj.transpose(-2,-1)