import numpy as np
import chess

def build_move_mapping():
    valids = set()
    for frsqr in range(64):
        for tosqr in range(64):
            fr, ff = frsqr % 8, frsqr >> 3
            tr, tf = tosqr % 8, tosqr >> 3
            dr, df = abs(tr - fr), abs(tf - ff)
            straight = tr == fr or tf == ff
            diag = dr == df
            horse = (dr == 2 and df == 1) or (dr == 1 and df == 2)
            if (straight or diag or horse) and frsqr != tosqr:
                valids.add((frsqr,tosqr,0))
    def addprom(frsqr, tosqr):
        valids.add((frsqr, tosqr, 1))
        valids.add((frsqr, tosqr, 2))
        valids.add((frsqr, tosqr, 3))
    for frsqr in range(48, 56):
        f = frsqr % 8
        addprom(frsqr, frsqr + 8)
        if f > 0: addprom(frsqr, frsqr + 7)
        if f < 7: addprom(frsqr, frsqr + 9)
    map = { (fr,to,p) : i for i, (fr,to,p) in enumerate(valids)}
    return map

def get_global_features(board, player):
    return np.array([
        board.has_kingside_castling_rights(player),
        board.has_queenside_castling_rights(player),
        board.has_kingside_castling_rights(not player),
        board.has_queenside_castling_rights(not player),
        board.ep_square + 1 if board.ep_square is not None else 0,
        board.has_legal_en_passant(),
        board.halfmove_clock,
        board.is_repetition(2),
        board.is_repetition(3)
    ])

def board_to_tensor(board, flip): # out(64,), square classes 0=none, 1-6=white, 7-12=black
    t = np.zeros(64)
    for square, piece in board.piece_map().items():
        color = piece.color if flip else not piece.color
        sq = square ^ 56 if flip else square
        t[sq] = color * 6 + piece.piece_type + 1
    return t

def uci_move_to_tensor(fen, uci, player):
    m = chess.Move.from_uci(uci)
    board = chess.Board(fen)
    flip = not player
    t = np.full((1858,), -1)
    for lm in board.generate_legal_moves(): t[move_to_idx(lm, flip)]=0
    t[move_to_idx(m, flip)]=1
    return t

map = build_move_mapping()
def move_to_idx(m:chess.Move, flip):
    fr = m.from_square ^ 56 if flip else m.from_square
    to = m.to_square ^ 56 if flip else m.to_square
    p = 0 if (not m.promotion or m.promotion == chess.QUEEN) else m.promotion - 1
    return map[(fr,to,p)]
