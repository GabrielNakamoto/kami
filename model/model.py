from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from chessformer.data import build_move_mapping
from tinygrad.nn import Embedding, Linear, RMSNorm
import math

map, _ = build_move_mapping()
policy_map = Tensor([fr_ * 64 + to_ for fr_, to_, _ in map], dtype=dtypes.int32)

def build_mixed_precision(params):
    for p in params:
        if p.dtype == dtypes.float32 and p.ndim >= 2:
            p.bf16 = p.cast(dtypes.bfloat16).realize()

class BF16Linear(Linear):
    def __call__(self, x):
        w = getattr(self.weight, "_bf16", None) or self.weight
        out = x.linear(w.T)
        return out + self.bias if self.bias is not None else out

class Smolgen:
    def __init__(self, dim, n_heads):
        self.n_heads = n_heads
        self.proj_in = BF16Linear(dim, 32, bias=False)
        self.extract = BF16Linear(64*32, 256)
        self.ln_extract = RMSNorm(256)
        self.proj_head = BF16Linear(256, 256*n_heads)
        self.ln_heads = RMSNorm(256*n_heads)
        self.scale = BF16Linear(256, 64*64)
        self.scale.weight.assign(Tensor.zeros_like(self.scale.weight))
        self.scale.bias.assign(Tensor.zeros_like(self.scale.bias))
    def __call__(self, x:Tensor):
        x = self.proj_in(x).reshape(-1, 64*32)
        x = self.ln_extract(self.extract(x).float()).cast(dtypes.bfloat16).swish()
        x = self.ln_heads(self.proj_head(x).float()).cast(dtypes.bfloat16).swish()
        x = x.reshape(-1, self.n_heads, 256)
        return self.scale(x).reshape(-1, self.n_heads, 64, 64)

class LeelaAttention:
    def __init__(self, dim, n_heads):
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.smolgen = Smolgen(dim, n_heads)
        self.qkv_proj = BF16Linear(dim, dim*3, bias=False)
        self.out_proj = BF16Linear(dim, dim, bias=False)
    def __call__(self, x:Tensor, dropout_p:float=0.0):
        B, seqln = x.shape[0], x.shape[1]
        xqkv = self.qkv_proj(x).reshape(B, seqln, 3, self.n_heads, self.head_dim)
        get = lambda n: xqkv[:,:,n].transpose(1,2)
        q, k, v = get(0), get(1), get(2) # (bchsz, n_heads, 64, head_dim)
        s = self.smolgen(x) # (bchsz, n_heads, 64, 64)
        logits = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        logits = logits + s
        probs = logits.float().softmax(-1).dropout(dropout_p)
        out = probs.cast(dtypes.bfloat16) @ v
        out = out.transpose(1, 2).reshape(B, seqln, self.dim)
        return self.out_proj(out)

class TransformerBlock:
    def __init__(self, dim, n_heads, use_lc_attn:bool=False):
        self.lcattn = LeelaAttention(dim, n_heads) if use_lc_attn else None
        self.dim = dim
        self.qkv_proj = None if use_lc_attn else BF16Linear(dim, dim*3, bias=False)
        self.attn_proj = None if use_lc_attn else BF16Linear(dim, dim, bias=False)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.ffn = [BF16Linear(dim, dim*2), Tensor.silu, BF16Linear(dim*2, dim)]
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
    def _attention(self, x: Tensor, dropout_p:float=0.0) -> Tensor: # x(bchsz, 64, dim)
        bchsz, seqln = x.shape[0], x.shape[1]
        xqkv = self.qkv_proj(x).float().reshape((bchsz, seqln, 3, self.n_heads, self.head_dim))
        getf = lambda n: xqkv[:,:,n].transpose(1,2)
        q, k, v = getf(0), getf(1), getf(2)
        attn = q.scaled_dot_product_attention(k, v, is_causal=False, dropout_p=dropout_p).transpose(2,1).reshape((bchsz, seqln, self.dim))
        return self.attn_proj(attn.cast(dtypes.bfloat16))
    def __call__(self, x: Tensor) -> Tensor:
        normed = self.attn_norm(x).cast(dtypes.bfloat16)
        if self.lcattn: x = x + self.lcattn(normed, dropout_p=0.05).float()
        else: x = x + self._attention(normed, dropout_p=0.05).float()
        return x + self.ffn_norm(x).cast(dtypes.bfloat16).sequential(self.ffn).dropout(0.05)

class Model:
    def __init__(self, dim:int, layers:int, n_heads:int, use_lc_attn:bool=False):
        self.dim = dim
        self.piece_emb = Embedding(13, dim)
        self.proj_glob = BF16Linear(9, dim)
        self.final_norm = RMSNorm(dim)
        self.blocks = [TransformerBlock(dim, n_heads, use_lc_attn=use_lc_attn) for _ in range(layers)]
        self.policy_from_proj = BF16Linear(dim, 64)
        self.policy_to_proj = BF16Linear(dim, 64)
    def __call__(self, pieces: Tensor, global_features: Tensor) -> Tensor:
        x = self.piece_emb(pieces) + self.proj_glob(global_features).unsqueeze(1)
        x = x.sequential(self.blocks)
        x = self.final_norm(x.float()).cast(dtypes.bfloat16)
        q = self.policy_from_proj(x)
        k = self.policy_to_proj(x)
        logits_4096 = (q @ k.transpose(-2, -1)).reshape(-1, 64*64)
        return logits_4096[:, policy_map].float()
