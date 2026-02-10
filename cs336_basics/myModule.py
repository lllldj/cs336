import torch
import torch.nn as nn
import os


class toy_Liner(nn.Module):
    def __init__(self, in_features, out_features, bias=None, device=None, dtype=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features,dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features,dtype=dtype)) if bias else None
        self.device = device
        
        self.set_weights()
    
    def set_weights(self,w=None):
        if w == None:
            nn.init.trunc_normal_(self.weight)
        else:
            self.weight.data = w
    
    def forward(self,x):
        out = x @ self.weight.T
        if self.bias != None:
            out += self.bias
        return out
    

class toy_Embedding(nn.Module):
    def __init__(self, num_embd, embd_dim, device = None,dtype = torch.float32) -> None:
        super().__init__()
        self.embd = nn.Parameter(torch.empty(num_embd,embd_dim,dtype=dtype))
        self.device = device

        self.set_embd()
        
    def set_embd(self,embd=None):
        if embd == None:
            nn.init.trunc_normal_(self.embd)
        else:
            self.embd.data = embd 
    
    def forward(self,x):
        return self.embd[x]
    
class toy_RMSnorm(nn.Module):
    def __init__(self, d_model, eps: float = 1e-5, device = None, dtype = torch.float32):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.empty(d_model,dtype=dtype))

        self.set_para()
        
    def set_para(self,g=None):
        if g==None:
            nn.init.trunc_normal_(self.gain,1,0.02)
        else:
            self.gain.data = g
    
    def forward(self,x):
        rmsx = x.square().mean(-1,keepdim=True) 
        out = x*self.gain/torch.sqrt(rmsx+self.eps)
        return out 
    
class toy_SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=torch.float32):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(d_ff,d_model,dtype=dtype))
        self.W2 = nn.Parameter(torch.empty(d_model,d_ff,dtype=dtype))
        self.W3 = nn.Parameter(torch.empty(d_ff,d_model,dtype=dtype))
    
        self.set_para()
    def set_para(self,w1=None,w2=None,w3=None):
        if w1 == None:
            nn.init.trunc_normal_(self.W1)
        else:
            self.W1.data = w1
        if w2 == None:
            nn.init.trunc_normal_(self.W2)
        else:
            self.W2.data = w2
        if w3 == None:
            nn.init.trunc_normal_(self.W3)
        else:
            self.W3.data = w3
    
    def forward(self,x):
        W3x = x @ self.W3.T
        W1x = x @ self.W1.T
        Slu = W1x * torch.sigmoid(W1x)
        return (Slu * W3x)@self.W2.T