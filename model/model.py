from tinygrad.tensor import Tensor
from tinygrad.nn import Embedding, Linear, RMSNorm
import math

class Smolgen:
    def __init__(self, dim, n_heads):
        self.n_heads = n_heads
        self.proj_in = Linear(dim, 32, bias=False)
        self.extract = Linear(64*32, 256)
        self.ln_extract = RMSNorm(256)
        self.proj_head = Linear(256, 256*n_heads)
        self.ln_heads = RMSNorm(256*n_heads)
        self.scale = Linear(256, 64*64)
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
        self.qkv_proj = Linear(dim, dim*3, bias=False)
        self.out_proj = Linear(dim, dim, bias=False)
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

class TransformerBlock:
    def __init__(self, dim, n_heads, use_lc_attn:bool=False):
        self.lcattn = LeelaAttention(dim, n_heads) if use_lc_attn else None
        self.dim = dim
        self.qkv_proj = None if use_lc_attn else Linear(dim, dim*3, bias=False)
        self.attn_proj = None if use_lc_attn else Linear(dim, dim, bias=False)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.ffn = [
            Linear(dim, dim*2),
            Tensor.silu,
            Linear(dim*2, dim)
        ]
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
    def _attention(self, x: Tensor, dropout_p:float=0.0) -> Tensor: # x(bchsz, 64, dim)
        bchsz, seqln = x.shape[0], x.shape[1]
        xqkv = self.qkv_proj(x).reshape((bchsz, seqln, 3, self.n_heads, self.head_dim))
        getf = lambda n: xqkv[:,:,n].transpose(1,2)
        q, k, v = getf(0), getf(1), getf(2)
        attn = q.scaled_dot_product_attention(k, v, is_causal=False, dropout_p=dropout_p).transpose(2,1).reshape((bchsz, seqln, self.dim))

        return self.attn_proj(attn)
    def __call__(self, x: Tensor) -> Tensor:
        if self.lcattn: x = x + self.lcattn(self.attn_norm(x), dropout_p=0.05)
        else: x = x + self._attention(self.attn_norm(x), dropout_p=0.05)
        return x + self.ffn_norm(x).sequential(self.ffn).dropout(0.05)

class Model:
    def __init__(self, dim:int, layers:int, n_heads:int, use_lc_attn:bool=False):
        self.dim = dim
        self.proj_in = Linear(15, dim)
        self.pos_emb = Embedding(64, dim)
        self.final_norm = RMSNorm(dim)
        self.blocks = [TransformerBlock(dim, n_heads, use_lc_attn=use_lc_attn) for _ in range(layers)]
        self.policy_head = Linear(dim, 73)

    def __call__(self, pieces: Tensor, global_features: Tensor) -> tuple[Tensor, Tensor]:
        B = pieces.shape[0]
        pt = pieces.reshape(B, 8, 64).transpose(1,2)
        g = global_features.unsqueeze(1).expand((B,64,-1)) # each square gets global information
        x = pt.cat(g, dim=-1)
        x = self.proj_in(x)
        x = x + self.pos_emb(Tensor.arange(64))
        x = x.sequential(self.blocks)
        x = self.final_norm(x)
        return self.policy_head(x).reshape(-1, 4672)

