from dataclasses import dataclass, field
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
    Nsa: dict = field(default_factory=dict)
    Ls: dict = field(default_factory=dict)
    Sum: dict = field(default_factory=dict)
    Es: dict = field(default_factory=dict)
    fpu: float = 0.0

    def update(self, s, a, v, getBatch, vl:int=1):
        if not getBatch and (v is not None):
            self.Nsa[s][a] += 1
            self.Sum[s][a] += v
        else:
            if v is None:
                mew = self.Sum[s][a] / self.Nsa[s][a] if self.Nsa[s][a] > 0 else self.fpu
                self.Nsa[s][a] += vl
                self.Sum[s][a] += vl * mew
            else:
                self.Nsa[s][a] += 1
                self.Sum[s][a] += v

# https://ludii.games/citations/ARXIV2021-1.pdf
class BatchMCTS:
    def __init__(self, model:Model, c_puct:float=1.2, FPU:float=0.25):
        self.model = model
        self.c_puct, self.fpu = c_puct, FPU
        self.batch_tree, self.tree = Tree(fpu=FPU), Tree(fpu=FPU)
        self.batch_boards: list[np.ndarray] = []
        self.batch_globals: list[np.ndarray] = []
        self.batch_fens: list[str] = []
        self.trans_table = {}
    def _puct(self, board:chess.Board, getBatch:bool):
        s = board._transposition_key()
        fen = board.fen()
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
                    self.batch_fens.append(fen)
                    self.batch_boards.append(board_to_tensor(board, flip))
                    self.batch_globals.append(get_global_features(board, board.turn))
                return None
            else:
                legals = list(board.generate_legal_moves())
                tree.Ls[s]=legals
                tree.Sum[s]=np.zeros(len(legals), dtype=np.float32)
                tree.Nsa[s]=np.zeros(len(legals), dtype=np.uint32)
                return self.trans_table[s][1]

        mew = np.where(tree.Nsa[s] > 0, tree.Sum[s] / tree.Nsa[s], self.fpu)
        bandit = mew + self.c_puct * self.trans_table[s][0] * math.sqrt(tree.Nsa[s].sum()) / (1. + tree.Nsa[s])
        puct = bandit.argmax(-1)
        board.push(tree.Ls[s][puct])

        v = self._puct(board, getBatch)
        tree.update(s, puct, v, getBatch)
        return -v if v is not None else None

    def _get_batch(self, fen:str, B:int):
        self.batch_boards, self.batch_globals, self.batch_fens, self.batch_tree = [], [], [], copy.deepcopy(self.tree)
        while len(self.batch_fens) < B: self._puct(chess.Board(fen), True)

    def _put_batch(self, fen:str, out:tuple[Tensor,Tensor]):
        pl, vl = out[0], out[1]
        probs, values = pl.softmax(axis=-1).numpy(), (vl.softmax(axis=-1) @ Tensor([1.,0.,-1.])).numpy()
        for i, lfen in enumerate(self.batch_fens):
            board = chess.Board(lfen)
            flip = not board.turn
            legals = list(board.generate_legal_moves())
            indices = [move_to_idx(lm, flip) for lm in legals]
            self.trans_table[board._transposition_key()]=(probs[i][indices],-values[i])
        i = 0
        while (v := self._puct(chess.Board(fen), False)) is not None:
            v = self._puct(chess.Board(fen), False)
            i += 1

    def search_iter(self, fen:str, num_batches:int=32, batch_size:int=32):
        self.tree = Tree()
        s = chess.Board(fen)._transposition_key()
        for n in range(num_batches):
            self._get_batch(fen, batch_size)
            x = Tensor(np.stack(self.batch_boards), dtype=dtypes.uint16)
            xg = Tensor(np.stack(self.batch_globals), dtype=dtypes.float32)
            out = self.model(x, xg)
            self._put_batch(fen, out)
            yield n, self.tree.Ls[s], self.tree.Nsa[s]

    def __call__(self, fen:str, num_batches:int=32, batch_size:int=32, second_move_heuristic:bool=True) -> tuple[chess.Move, np.ndarray]:
        s = chess.Board(fen)._transposition_key()
        for n, _, nsa in self.search_iter(fen, num_batches, batch_size):
            print(n, nsa)
        best = self.tree.Nsa[s].argmax(-1)
        return self.tree.Ls[s][best], self.tree.Nsa[s]
