import chess, numpy as np, os
from tqdm import tqdm
from datasets import load_dataset

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
    imap = { v : k for k, v in map.items() }
    return map, imap

def fen_to_tensor(fen, flip): # out(64,), square classes 0=none, 1-6=white, 7-12=black
    board = chess.Board(fen)
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

map, imap = build_move_mapping()
def move_to_idx(m:chess.Move, flip):
    fr = m.from_square ^ 56 if flip else m.from_square
    to = m.to_square ^ 56 if flip else m.to_square
    p = 0 if (not m.promotion or m.promotion == chess.QUEEN) else m.promotion - 1
    return map[(fr,to,p)]

def tensorize_batch(batch):
    xs, yzs, yps = [], [], []
    for hist, moves, z in zip(batch['fen_history'], batch['uci_moves'], batch['z']):
        player = hist[-1].split()[-5] == 'w'
        xs.append(np.stack([fen_to_tensor(f, not player) for f in hist]))
        yzs.append(np.eye(3, dtype=np.float32)[z])
        yps.append(uci_move_to_tensor(hist[-1], moves[-1], player))
    return xs, yzs, yps

OUT_DIR = "tensors"
os.makedirs(OUT_DIR, exist_ok=True)
dataset = load_dataset("gRa1ne/decorrelated-chess-3.8m", split='train')
N = len(dataset)

x = np.memmap(f"{OUT_DIR}/x.bin", dtype=np.uint8, mode="w+", shape=(N, 8, 64))
yz = np.memmap(f"{OUT_DIR}/yz.bin", dtype=np.float32, mode="w+", shape=(N, 3))
yp = np.memmap(f"{OUT_DIR}/yp.bin", dtype=np.int8, mode="w+", shape=(N, 1858))

for start in tqdm(range(8, N, 512)):
    end = min(start + 512, N)
    xs, yzs, yps = tensorize_batch(dataset[start:end])
    x[start:end]=np.stack(xs).astype(np.uint8)
    yz[start:end]=np.stack(yzs)
    yp[start:end]=np.stack(yps).astype(np.int8)
