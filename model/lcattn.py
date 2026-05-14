from tinygrad.tensor import Tensor
import math
from tinygrad import nn


class Smolgen:
    def __init__(self, dim, n_heads):
        self.n_heads = n_heads
        self.proj_in = nn.Linear(dim, 32, bias=False)
        self.extract = nn.Linear(64*32, 256)
        self.ln_extract = nn.LayerNorm(256)
        self.proj_head = nn.Linear(256, 256*n_heads)
        self.ln_heads = nn.LayerNorm(256*n_heads)
        self.scale = nn.Linear(256, 64*64)
        self.scale.weight = Tensor.zeros(64*64, 256)
        self.scale.bias = Tensor.zeros(64*64)
    def __call__(self, x:Tensor):
        x = self.proj_in(x).reshape(-1, 64*32)
        x = self.ln_extract(self.extract(x)).swish()
        x = self.ln_heads(self.proj_head(x)).swish()
        x = x.reshape(-1, self.n_heads, 256)
        return self.scale(x).reshape(-1, self.n_heads, 64, 64)

class LeelaAttention:
    def __init__(self, dim, n_heads):
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.smolgen = Smolgen(dim, n_heads)
        self.qkv_proj = nn.Linear(dim, dim*3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
    def __call__(self, x:Tensor, dropout_p:float=0.0):
        B, seqln = x.shape[0], x.shape[1]
        xqkv = self.qkv_proj(x).reshape(B, seqln, 3, self.n_heads, self.head_dim)
        get = lambda n: xqkv[:,:,n].transpose(1,2)
        q, k, v = get(0), get(1), get(2) # (bchsz, n_heads, 64, head_dim)
        s = self.smolgen(x) # (bchsz, n_heads, 64, 64)
        attn = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        out = (attn + s).softmax(-1).dropout(dropout_p) @ v
        out = out.transpose(1, 2).reshape(B, seqln, self.dim)
        return self.out_proj(out)
