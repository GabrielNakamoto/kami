from multiprocessing import Pool
from functools import partial
import pyarrow as pa, pyarrow.parquet as pq
import chess, random, argparse
from stockfish import Stockfish
from tqdm import tqdm

def init_worker(fp):
    global stockfish
    stockfish = Stockfish(fp)

SAMPLES_PER_GAME = 3
def process_batch(batch):
    hs, zs, ms = [], [], []
    for game in batch:
        game = game.split()
        moves = [m.lower() for m in game[:-1]]
        if len(moves) < 15: continue
        board = chess.Board()
        ts = random.sample(range(6, len(moves)-1), SAMPLES_PER_GAME)
        for i in range(ts[-1]+1):
            board.push_uci(moves[i])
            if i in ts:
                stockfish.set_fen_position(board.fen())
                wdl = [x/1000. for x in stockfish.get_wdl_stats()]
                zs.append(wdl)
                hs.append(board.fen())
                ms.append(moves[i+1])
    return hs, zs, ms

SCHEMA = pa.schema([("fens", pa.string()), ("move_played", pa.string()), ("stockfish_wdl", pa.list_(pa.float32(), 3))])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sfpath", type=str, help="path to local stockfish executable for wdl injection")
    cfg = parser.parse_args()
    games = [l for l in open("raw/raw.uci").read().splitlines() if l.strip()]
    bchsz = 512
    N = len(games) // bchsz
    batches = [games[i*bchsz:(i*bchsz)+bchsz] for i in range(N)]
    with pq.ParquetWriter("data.parquet", SCHEMA, compression="snappy") as writer:
        with Pool(initializer=partial(init_worker, cfg.sfpath)) as pool:
            for hs, zs_batch, ms_batch in tqdm(pool.imap_unordered(process_batch, batches, chunksize=1), total=len(batches), unit="batch"):
                writer.write_table(pa.table({"fens": hs, "move_played": ms_batch, "stockfish_wdl": zs_batch}, schema=SCHEMA))
