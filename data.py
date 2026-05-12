import chess, sys
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login
from multiprocessing import Pool

QUEEN_DIRS = {(1,0):0, (1,1):1, (1,-1):2, (-1,0):3, (-1,1):4, (-1,-1):5, (0,1):6, (0,-1):7}
KNIGHT_DIRS = {(2,1):0, (2,-1):1, (1,2):2, (1,-2):3, (-1,2):4, (-1,-2):5, (-2,1):6, (-2,-1):7}
PROMO_IDX = {chess.KNIGHT:0, chess.BISHOP:1, chess.ROOK:2}

def move_plane(from_sq, to_sq, promotion):
    fr, ff = from_sq >> 3, from_sq & 7
    tr, tf = to_sq >> 3, to_sq & 7
    dr, df = tr - fr, tf - ff
    if dr == 0 or df == 0 or abs(dr) == abs(df):
        mag = df if dr == 0 else dr
        sdr = (dr > 0) - (dr < 0)
        sdf = (df > 0) - (df < 0)
        n = QUEEN_DIRS[sdr, sdf] * 7 + abs(mag) - 1
    elif promotion and promotion != chess.QUEEN:
        n = 64 + PROMO_IDX[promotion] * 3 + (df + 1)
    else:
        n = 56 + KNIGHT_DIRS[dr, df]
    return (fr * 8 + ff) * 73 + n

def encode_pieces_into(out, board):
    out.fill(0)
    pm = board.piece_map()
    if board.turn == chess.WHITE:
        for sq, p in pm.items():
            out[sq] = p.piece_type + (p.color * 6)
    else:
        # flip vertically and swap colors
        for sq, p in pm.items():
            out[(7 - (sq >> 3)) * 8 + (sq & 7)] = p.piece_type + ((not p.color) * 6)

def encode_game(game):
    winner, uci_moves = game['winner'], game['moves_uci']
    z = 0 if not winner else 1 if winner == 'white' else -1
    board = chess.Board()
    n = len(uci_moves)
    P = np.zeros((n, 8, 64), dtype=np.int8)
    G = np.zeros((n, 7), dtype=np.int16)
    M = np.full((n, 4672), -1, dtype=np.int8)
    O = np.zeros(n, dtype=np.int8)

    # circular 8-ply history written directly into P[i]
    hist = np.zeros((8, 64), dtype=np.int8)
    hist_filled = 0
    encode_pieces_into(hist[7], board)
    hist_filled = 1

    rep = {}
    rep[board._transposition_key()] = 1

    for i, uci in enumerate(uci_moves):
        # piece tensor: last 8 plies, zero-padded on the left
        if hist_filled < 8:
            P[i, 8 - hist_filled:] = hist[8 - hist_filled:]
        else:
            P[i] = hist

        white_turn = board.turn == chess.WHITE
        key = board._transposition_key()
        rcount = rep.get(key, 1)
        G[i, 0] = white_turn
        G[i, 1] = board.has_kingside_castling_rights(board.turn)
        G[i, 2] = board.has_queenside_castling_rights(board.turn)
        G[i, 3] = board.has_kingside_castling_rights(not board.turn)
        G[i, 4] = board.has_queenside_castling_rights(not board.turn)
        G[i, 5] = min(board.halfmove_clock, 100)
        G[i, 6] = 2 if rcount >= 3 else (1 if rcount == 2 else 0)

        row = M[i]
        flip = not white_turn
        for lm in board.generate_legal_moves():
            fs, ts = lm.from_square, lm.to_square
            if flip:
                fs = (7 - (fs >> 3)) * 8 + (fs & 7)
                ts = (7 - (ts >> 3)) * 8 + (ts & 7)
            row[move_plane(fs, ts, lm.promotion)] = 0

        m = chess.Move.from_uci(uci)
        fs, ts = m.from_square, m.to_square
        if flip:
            fs = (7 - (fs >> 3)) * 8 + (fs & 7)
            ts = (7 - (ts >> 3)) * 8 + (ts & 7)
        row[move_plane(fs, ts, m.promotion)] = 1

        O[i] = z if white_turn else -z

        board.push(m)
        hist[:-1] = hist[1:]
        encode_pieces_into(hist[-1], board)
        hist_filled = min(hist_filled + 1, 8)

        k = board._transposition_key()
        rep[k] = rep.get(k, 0) + 1

    return P.reshape(n, 512), G, M, O

if __name__ == "__main__":
    login()
    N = int(sys.argv[1])
    pieces = np.lib.format.open_memmap("pieces.np", shape=(N,512), dtype=np.int8, mode="w+")
    globs = np.lib.format.open_memmap("globals.np", shape=(N,7), dtype=np.int16, mode="w+")
    moves = np.lib.format.open_memmap("moves.np", shape=(N,4672), dtype=np.int8, mode="w+")
    outcomes = np.lib.format.open_memmap("outcomes.np", shape=(N,), dtype=np.int8, mode="w+")
    ds = load_dataset('angeluriot/chess_games', streaming=True)['train']
    i = 0
    bar = tqdm(total=N, unit="move")
    with Pool() as pool:
        for P,G,M,O in pool.imap_unordered(encode_game, ds, chunksize=8):
            n = min(len(P), N-i)
            pieces[i:i+n]=P[:n]
            globs[i:i+n]=G[:n]
            moves[i:i+n]=M[:n]
            outcomes[i:i+n]=O[:n]
            i += n
            bar.update(n)
            if i == N: break
