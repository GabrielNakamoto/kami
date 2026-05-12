from tinygrad.tensor import Tensor
from tinygrad.nn import Embedding, Linear, RMSNorm

class TransformerBlock:
    def __init__(self, dim, n_heads):
        self.dim = dim
        self.qkv_proj = Linear(dim, dim*3)
        self.attn_proj = Linear(dim, dim)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.ffn = [
            Linear(dim, dim*4),
            Tensor.silu,
            Linear(dim*4, dim)
        ]
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
    def _attention(self, x: Tensor) -> Tensor: # x(bchsz, 64, dim)
        bchsz, seqln = x.shape[0], x.shape[1]
        xqkv = self.qkv_proj(x).reshape((bchsz, seqln, 3, self.n_heads, self.head_dim))
        getf = lambda n: xqkv[:,:,n].transpose(1,2)
        q, k, v = getf(0), getf(1), getf(2)
        attn = q.scaled_dot_product_attention(k, v, is_causal=False).transpose(2,1).reshape((bchsz, seqln, self.dim))
        return self.attn_proj(attn)
    def __call__(self, x: Tensor) -> Tensor:
        x = x + self._attention(self.attn_norm(x)).dropout(0.1)
        return x + self.ffn_norm(x).sequential(self.ffn).dropout(0.1)

class Model:
    def __init__(self, dim: int, layers: int, n_global, n_heads: int):
        self.dim = dim
        self.pos_emb = Embedding(64, dim)
        self.piece_emb = Embedding(13, dim)
        self.ply_emb = Embedding(8, dim)
        self.ply_project = Linear(dim*8, dim)
        self.global_proj = [
            Linear(n_global, dim),
            Tensor.silu,
            Linear(dim, dim)
        ]
        self.final_norm = RMSNorm(dim)
        self.blocks = [TransformerBlock(dim, n_heads) for _ in range(layers)]
        self.policy_head = Linear(dim, 73)

        self.value_proj = Linear(dim, 1)

    def __call__(self, pieces: Tensor, global_features: Tensor) -> tuple[Tensor, Tensor]:
        B = pieces.shape[0]
        pieces = self.piece_emb(pieces.reshape(B,8,64))                     # (B,8,64,dim)
        pieces = pieces + self.ply_emb(Tensor.arange(8)).reshape(1,8,1,-1)  # (B,8,64,dim)
        pieces = pieces.permute(0,2,1,3).reshape(B,64,8*self.dim)           # (B,64,8*dim)
        pieces = self.ply_project(pieces)                                   # (B,64,dim)
        pieces = pieces + self.pos_emb(Tensor.arange(64))
        global_features = global_features.sequential(self.global_proj).unsqueeze(1)
        x = global_features.cat(pieces, dim=1) # x(batch, 65, dim)
        x = x.sequential(self.blocks)
        x = self.final_norm(x)
        squares = x[:,1:,:]
        global_tok = x[:,0,:]

        p = self.policy_head(squares).reshape(-1, 4672)
        v = self.value_proj(global_tok).tanh().squeeze(-1)
        return p, v

