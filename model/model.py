from tinygrad.tensor import Tensor
from tinygrad.nn import Embedding, Linear, RMSNorm
from .lcattn import LeelaAttention

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

        self.value_proj = [
            Linear(dim, 128),
            Tensor.relu,
            Linear(128, 1)
        ]

    def __call__(self, pieces: Tensor, global_features: Tensor) -> tuple[Tensor, Tensor]:
        B = pieces.shape[0]
        pt = pieces.reshape(B, 8, 64).transpose(1,2)
        g = global_features.unsqueeze(1).expand((B,64,-1)) # each square gets global information
        x = pt.cat(g, dim=-1)
        x = self.proj_in(x)
        x = x + self.pos_emb(Tensor.arange(64))
        x = x.sequential(self.blocks)
        x = self.final_norm(x)
        # include square info for value head
        vx = x.mean(axis=1)
        p = self.policy_head(x).reshape(-1, 4672)
        v = vx.sequential(self.value_proj).tanh().squeeze(-1)
        return p, v

