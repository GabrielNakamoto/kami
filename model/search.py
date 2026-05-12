from __future__ import annotations
import copy
import numpy as np
import chess

def search():
    pass

"""
Ns = # times state visited, cache for upper bound calculation
Nsa = # times edge visited
priors = model prediction of taking an edge
Qsa = expected reward/win rate of taking an edge based on search frequency and model value head

u = upper confidence bound, used to decide where to explore next
    - depends on Qsa (expected) + c_pcut (exploration constant) * policy head output 
        and exploration weighted on how many times been visited relative to parent state of edge

A inference time pick edge from current state with greatest visits?

best_action = max(Nsa[s].values())
"""

"""


Nsa, Qsa, priors, Ns = {}, {}, {}, {}

def search(state:GameState, model:Model, player: chess.Color, cpuct:float=0.1):
    s = str(state)
    if state.board.is_game_over():
        if outcome := state.board.outcome():
            winner = outcome.winner
            if not winner: return 1e-6
            elif winner == player: return 1.0
            else: return -1.0

    if s not in priors:
        priors[s], v = model(state.piece_tensor(), state.global_tensor())
        Ns[s] = 0
        return -v

    # Pick best upper bound confidence
    best = (-float("inf"), -1)
    valids = list(state.board.legal_moves)
    for a in range(len(valids)):
        if (s, a) in Qsa:
            # doesnt work right now, priors[s][a], policy output Tensor cant be indexed like that
            u = Qsa[(s,a)] + cpuct * priors[s][a] * np.sqrt(Ns[s]) / (1 + Nsa[s][a])
        else:
            u = cpuct * priors[s][a] * np.sqrt(Ns[s] + 1e-8)
        if u > best[0]: best = (u,a)

    a = best[1]
    next_s = copy.copy(state).push(valids[a])
    v = search(next_s, model, player, cpuct=cpuct)
    
    if (s, a) in Qsa:
        Qsa[(s,a)]=(Nsa[(s,a)] * Qsa[(s,a) + v] / (Nsa[(s,a)] + 1))
        Nsa[s][a]+=1
    else:
        Qsa[(s,a)]=v
        Nsa[s][a]=1
    Ns[s] += 1
    return -v
"""
