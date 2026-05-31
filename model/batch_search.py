from dataclasses import dataclass
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
import numpy as np, math, copy
from util.convert import board_to_tensor, get_global_features, move_to_idx
from model import Model
import chess

# We rollout/explore simulations to build a tree from current game state.
# Nodes are states and directed edges are legal actions
# Q = expected reward for taking edge
# N = # of times edge/node has been visited
# P = policy prior, distribution over actions from a given state
# L = legal move indices
# For each (s,a) dict represent the enumeration over actions as a np array for efficient puct computation

@dataclass
class Tree:
    Nsa: dict = {}
    Qsa: dict = {}
    Ls: dict = {}
    Es: dict = {}
    Sum: dict = {}

    def update(self, fen:str, a, v, getBatch, vl:float=0.0):
        board = chess.Board(fen)
        s = board._transposition_key()
        if not getBatch and (v is not None):
            self.Nsa[s][a] += 1
            self.Sum[s][a] += v
        else:
            if v is None:
                mew = self.Sum[s][a] / self.Nsa[s][a]
                self.Nsa[s][a] += vl
                self.Sum[s][a] += vl * mew
            else:
                self.Nsa[s][a] += 1
                self.Sum[s][a] += v

# https://ludii.games/citations/ARXIV2021-1.pdf
class BatchMCTS:
    def __init__(self, model:Model, c_puct:float=1.0, FPU:float=0.0):
        self.model = model
        self.c_puct, self.fpu = c_puct, FPU
        self.batch_tree, self.tree = Tree(), Tree()
        self.batch: list[tuple[Tensor, Tensor]] = []
        self.trans_table = {}
    def _puct(self, fen:str, getBatch:bool):
        board = chess.Board(fen)
        s = board._transposition_key()
        tree = self.batch_tree if getBatch else self.tree

        if s in tree.Es:
            return -tree.Es[s]

        if board.is_game_over(): 
            r, z = board.result(), 0.0
            if r == "1-0": z = 1.0
            elif r == "0-1": z = -1.0
            if board.turn != chess.WHITE: z = -z
            tree.Es[s] = z
            return -z

        if s not in tree.Nsa:
            flip = not board.turn
            if s not in self.trans_table:
                if getBatch:
                    self.batch.append((
                        Tensor(board_to_tensor(board, flip), dtype=dtypes.uint16),
                        Tensor(get_global_features(board, board.turn), dtype=dtypes.float32).unsqueeze(0)
                    ))
                return None
            else:
                # add s to t
                legals = list(board.generate_legal_moves())
                tree.Ls[s]=legals
                tree.Qsa[s]=np.zeros(len(legals), dtype=np.float32)
                tree.Nsa[s]=np.zeros(len(legals), dtype=np.uint32)
                return self.trans_table[s][1]

        mew = (tree.Nsa[s] > 0).where(tree.Sum[s] / tree.Nsa[s], self.fpu)
        bandit = mew + self.c_puct * self.trans_table[s][0] * math.sqrt(tree.Nsa[s].sum()) / (1. + tree.Nsa[s])
        puct = bandit.argmax(-1)
        board.push(tree.Ls[s][puct])

        next_fen = board.fen()
        v = self._puct(next_fen, getBatch)
        tree.update(fen, puct, v, getBatch)
        return -v if v else None

    def _get_batch(self, fen:str, B:int):
        self.batch, self.batch_tree = [], copy.deepcopy(self.tree)
        while len(self.batch) < B: self._puct(fen, True)

    def _put_batch(self, fen:str, out:tuple[Tensor,Tensor]):
        board = chess.Board(fen)
        ps, vs = out[0], out[1]
        for pl, vl in zip(ps.split(1,dim=0), vs.split(1,dim=0)):
            flip = not board.turn
            legals = list(board.generate_legal_moves())
            indices = [move_to_idx(lm, flip) for lm in legals]
            policy = pl.flatten()[indices].softmax().numpy()
            value = -vl.softmax().dot(Tensor([1.0, 0.0, -1.0])).item()
            self.trans_table[board._transposition_key()]=(policy,value)
            return -vl.softmax().dot(Tensor([1.0, 0.0, -1.0])).item()

    def __call__(self, fen:str, num_batches:int, batch_size:int) -> chess.Move:
        self.tree = Tree()
        for _ in range(num_batches):
            self._get_batch(fen, batch_size)
            x, xg = Tensor.stack(*[b[0] for b in self.batch]), Tensor.stack(*[b[1] for b in self.batch])
            out = self.model(x, xg)
            self._put_batch(fen, out)
        s = chess.Board(fen)._transposition_key()
        best = self.tree.Nsa[s].argmax(-1)
        return self.tree.Ls[s][best]
