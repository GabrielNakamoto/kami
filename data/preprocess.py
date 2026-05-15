from multiprocessing import Pool
import pyarrow as pa, pyarrow.parquet as pq
import chess, random
from tqdm import tqdm

# multiprocess-parallelism, batch games so each IPC overhead computes many games at once
def process_batch(batch):
    hs, zs, ms = [], [], []
    for game in batch:
        game = game.split()
        moves, result = [m.lower() for m in game[:-1]], game[-1]
        if len(moves) < 15: continue
        z = 0 if result == "0-1" else 2 if result == "1-0" else 1
        # skip first 6 + need 7 ply history
        board = chess.Board()
        n = random.randint(14, len(moves)-1)
        for i in range(n-8): board.push_uci(moves[i])
        history = []
        for i in range(n-8, n):
            board.push_uci(moves[i])
            history.append(board.fen())
        ms.append(moves[n+1])
        hs.append(history)
        zs.append(z)
    return hs, zs, ms

SCHEMA = pa.schema([
    ("fen_history", pa.list_(pa.string())),
    ("uci_moves",   pa.list_(pa.string())),
    ("z",           pa.int64()),
])

if __name__ == "__main__":
    games = [l for l in open("processed.uci").read().splitlines() if l.strip()]
    N = len(games) // 512
    batches = [games[i*512:(i*512)+512] for i in range(N)]
    with pq.ParquetWriter("data.parquet", SCHEMA, compression="snappy") as writer:
        with Pool() as pool:
            for hs, zs_batch, ms_batch in tqdm(pool.imap_unordered(process_batch, batches, chunksize=1), total=len(batches), unit="batch"):
                writer.write_table(pa.table({"fen_history": hs, "uci_moves": ms_batch, "z": zs_batch}, schema=SCHEMA))
