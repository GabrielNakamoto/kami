from tinygrad.tensor import Tensor
import numpy as np
import math
from util.convert import board_to_tensor, get_global_features, build_move_mapping
from model import Model
import chess

# We rollout/explore simulations to build a tree from current game state.
# Nodes are states and directed edges are legal actions
# Q = expected reward for taking edge
# N = # of times edge/node has been visited
# P = policy prior, distribution over actions from a given state
# L = legal move indices
# For each (s,a) dict represent the enumeration over actions as a np array for efficient puct computation
move_map = build_move_mapping()
inv_move_map = { v:k for k, v in move_map.items() }

class MCTS:
    def __init__(self, model:Model, c_puct:float=1.0):
        self.model = model
        self.c_puct = c_puct
        self.Qsa, self.Nsa, self.Ps, self.Ls, self.Es = {}, {}, {}, {}, {}

    def __call__(self, fen:str, num_sims:int=5):
        self.Qsa, self.Nsa, self.Ps, self.Ls, self.Es = {}, {}, {}, {}, {}
        for _ in range(num_sims): self.sim(fen)
        s = chess.Board(fen)._transposition_key()
        a = self.Nsa[s].argmax(-1)
        return chess.Move(*inv_move_map[self.Ls[s][a]])

    # https://suragnair.github.io/posts/alphazero.html
    def sim(self, fen:str):
        board = chess.Board(fen)
        s = board._transposition_key()

        if s in self.Es:
            return -self.Es[s]

        if board.is_game_over(): 
            r, z = board.result(), 0.0
            if r == "1-0": z = 1.0
            elif r == "0-1": z = -1.0
            if board.turn != chess.WHITE: z = -z
            self.Es[s] = z
            return -z

        if s not in self.Ps:
            pl, vl = self.model(
                Tensor(board_to_tensor(board, not board.turn)),
                Tensor(get_global_features(board, board.turn))
            )
            indices = [move_map[(lm.from_square, lm.to_square, lm.promotion)] for lm in board.generate_legal_moves()]
            self.Ls[s]=indices
            self.Ps[s]=pl.flatten()[indices].softmax().numpy()
            self.Qsa[s]=np.zeros(len(indices), dtype=np.float32)
            self.Nsa[s]=np.zeros(len(indices), dtype=np.uint32)
            return -vl.dot([1.0, 0.0, -1.0]).item()

        best = (self.Qsa[s] + self.c_puct * self.Ps[s] * math.sqrt(max(self.Nsa[s].sum(), 1)) / (1. + self.Nsa[s])).argmax(-1)
        board.push(chess.Move(*inv_move_map[self.Ls[s][best]]))

        next_fen = board.fen()
        v = self.sim(next_fen)

        self.Qsa[s][best] = (self.Nsa[s][best]*self.Qsa[s][best] + v) / (self.Nsa[s][best] + 1)
        self.Nsa[s][best] += 1
        return -v
