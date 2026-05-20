from multiprocessing import Pool
import pyarrow as pa, pyarrow.parquet as pq
import chess, random
from tqdm import tqdm

# multiprocess-parallelism, batch games so each IPC overhead computes many games at once
SAMPLES_PER_GAME = 3
def process_batch(batch):
    hs, zs, ms = [], [], []
    for game in batch:
        game = game.split()
        moves, result = [m.lower() for m in game[:-1]], game[-1]
        if len(moves) < 15: continue
        z = 0 if result == "1-0" else 2 if result == "0-1" else 1
        board = chess.Board()
        ts = sorted(random.sample(range(6, len(moves)-1), SAMPLES_PER_GAME))
        history, povs = [], []
        for i in range(ts[-1]+1):
            board.push_uci(moves[i])
            if i in ts:
                history.append(board.fen())
                if board.turn or z == 1: povs.append(z)
                else: povs.append(2 if not z else 0)
        ms.extend([moves[t+1] for t in ts])
        hs.extend(history)
        zs.extend(povs)
    return hs, zs, ms

SCHEMA = pa.schema([("fens", pa.string()), ("move_played", pa.string()), ("wdl", pa.int64()),])

if __name__ == "__main__":
    games = [l for l in open("raw/raw.uci").read().splitlines() if l.strip()]
    bchsz = 512
    N = len(games) // bchsz
    batches = [games[i*bchsz:(i*bchsz)+bchsz] for i in range(N)]
    with pq.ParquetWriter("data.parquet", SCHEMA, compression="snappy") as writer:
        with Pool() as pool:
            for hs, zs_batch, ms_batch in tqdm(pool.imap_unordered(process_batch, batches, chunksize=1), total=len(batches), unit="batch"):
                writer.write_table(pa.table({"fens": hs, "move_played": ms_batch, "wdl": zs_batch}, schema=SCHEMA))

