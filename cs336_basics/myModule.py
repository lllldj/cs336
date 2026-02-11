import torch
import torch.nn as nn
import os
import re
from collections import defaultdict

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
        out = x @ self.weight.transpose(-2,-1)
        if self.bias != None:
            out += self.bias
        return out
    

class toy_Embedding(nn.Module):
    def __init__(self, num_embd, embd_dim, device = None,dtype = torch.float32) -> None:
        super().__init__()
        self.embd = nn.Parameter(torch.empty(num_embd,embd_dim,dtype=dtype))
        self.device = device

        self.set_para()
        
    def set_para(self,embd=None):
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
        W3x = x @ self.W3.transpose(-2,-1)
        W1x = x @ self.W1.transpose(-2,-1)
        Slu = W1x * torch.sigmoid(W1x)
        return (Slu * W3x)@self.W2.transpose(-2,-1)
    
    
class toy_RoPE(nn.Module):
    def __init__(self, d_k, theta, max_len, device = None, dtype = torch.float32):
        super().__init__()
        
        self.rot_d = d_k//2
        i = torch.arange(self.rot_d, device=device, dtype=dtype)         
        j = torch.arange(max_len, device=device, dtype=dtype)      

        inv_freq = torch.exp(-(2*i)/d_k * torch.log(torch.tensor(theta, device=device, dtype=dtype)))                   
        thetas = j[:, None] * inv_freq[None, :]  
        
        cos_table = torch.cos(thetas)  #cos_table [token posistion, feature posistion]
        sin_table = torch.sin(thetas)
        
        self.register_buffer("cos_table",cos_table,persistent=False)
        self.register_buffer("sin_table",sin_table,persistent=False)
    
    def forward(self,x,tk_posistions):
        cos = self.cos_table[tk_posistions] #(T,d/2)
        sin = self.sin_table[tk_posistions] #(T,d/2)
        x_rot = x[..., :2*self.rot_d]
        x_pass = x[..., 2*self.rot_d:]
        x1 = x_rot[...,0::2] #(T,d/2 + 1) ?
        x2 = x_rot[...,1::2]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        y_rot = torch.stack([y1, y2], dim=-1).flatten(-2)
        return torch.cat([y_rot, x_pass], dim=-1)



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


    
class multi_attention(nn.Module):
    def __init__(self, d_in, num_heads, max_seq_len, theta, device = None) -> None:
        super().__init__()
        self.c_attention = toy_Liner(d_in, 3* d_in)
        self.proj = toy_Liner(d_in,d_in)
        self.num_head = num_heads
        self.d_head = d_in//self.num_head       
        self.ropez = toy_RoPE(self.d_head, theta ,max_seq_len)
        self.device = device
        
        trill_mask = torch.tril(torch.ones(max_seq_len,max_seq_len,dtype=torch.bool))
        self.register_buffer("trill",trill_mask,persistent=False)
    
    
    def forward(self,x):
        B, T, C = x.shape
        qkv = self.c_attention(x) #B,T,C @ C 3C -> B,T,3C
        Q,K,V = qkv.split(C,-1) # B,T,C
        
        qs = Q.view(B,T,self.num_head,self.d_head).transpose(1,2)
        ks = K.view(B,T,self.num_head,self.d_head).transpose(1,2) #B,h,T,d_h
        vs = V.view(B,T,self.num_head,self.d_head).transpose(1,2)
        
        tk_ps = torch.arange(T)
        qs = self.ropez.forward(qs,tk_ps)  
        ks = self.ropez.forward(ks,tk_ps)
        
        atts = toy_product_atte(qs,ks,vs,self.trill[:T,:T]).transpose(1, 2).contiguous().view(B,T,C) # B, T ,C
        return self.proj(atts)
    
class transformer_block(nn.Module):
    def __init__(self, d_in, num_heads, d_ff, max_seq_len, theta, device=None) -> None:
        super().__init__()   
        self.norm1 = toy_RMSnorm(d_in)
        self.atte = multi_attention(d_in,num_heads,max_seq_len,theta,device)
        self.norm2 = toy_RMSnorm(d_in)
        self.ff = toy_SwiGLU(d_in,d_ff)
        self.device = device
    
    def set_para(self,para_dict):
        q_proj_weight = para_dict["attn.q_proj.weight"]
        k_proj_weight = para_dict["attn.k_proj.weight"]
        v_proj_weight = para_dict["attn.v_proj.weight"]
        o_proj_weight = para_dict["attn.output_proj.weight"]
        ln1_weight = para_dict["ln1.weight"]
        ln2_weight = para_dict["ln2.weight"]
        ff_w1 = para_dict["ffn.w1.weight"]
        ff_w2 = para_dict["ffn.w2.weight"]
        ff_w3 = para_dict["ffn.w3.weight"]
        c_atte_weight = torch.cat([q_proj_weight,k_proj_weight,v_proj_weight],0)
        self.atte.c_attention.set_weights(c_atte_weight)
        self.atte.proj.set_weights(o_proj_weight)
        self.norm1.set_para(ln1_weight)
        self.norm2.set_para(ln2_weight)
        self.ff.set_para(ff_w1,ff_w2,ff_w3)
        
    def forward(self,x):
        x = x + self.atte(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x
    
    
    

class toy_Transformer_lm(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device = None) -> None:
        super().__init__()
        self.tk_embd = toy_Embedding(vocab_size,d_model)
        self.blocks =nn.ModuleList([transformer_block(d_model,num_heads,d_ff,context_length,rope_theta) for _ in range(num_layers)])
        self.norm = toy_RMSnorm(d_model)
        self.out_embd = toy_Liner(d_model,vocab_size)
    
    def set_para(self,para_dict):
        self.tk_embd.set_para(para_dict["token_embeddings.weight"])
        self.out_embd.set_weights(para_dict["lm_head.weight"])
        self.norm.set_para(para_dict["ln_final.weight"])
        grouped = defaultdict(dict)
        pat = re.compile(r"^layers\.(\d+)\.(.+)$")  # layers.{i}.rest

        for k, v in para_dict.items():
            m = pat.match(k)
            if m:
                i = int(m.group(1))
                rest = m.group(2)  # e.g. "ffn.w3.weight"
                grouped[i][rest] = v
        layer_dict = dict(grouped)
        for _ ,blk in enumerate(self.blocks):
            blk.set_para(layer_dict[_])
    def forward(self,x):
        #x : ids
        x = self.tk_embd(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.out_embd(x)