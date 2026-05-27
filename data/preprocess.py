from multiprocessing import Pool
import pyarrow as pa, pyarrow.parquet as pq
import chess, random
from stockfish import Stockfish
from tqdm import tqdm

sf_config = {
    "depth":8,
    "threads":1,
    "hash":128
}

def init_worker():
    global stockfish
    stockfish = Stockfish(
        depth=sf_config["depth"],
        parameters={
            "Threads" : sf_config['threads'],
            "Hash" : sf_config['hash'],
        }
    )

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

SCHEMA = pa.schema(
    [("fen_position", pa.string()),
     ("uci_move_played", pa.string()),
     ("stockfish_wdl", pa.list_(pa.float32(), 3))]
).with_metadata({"stockfish_wdl:" : f"Stockfish value injection trained with settings:\n{",".join([f"{k}={v}" for k, v in sf_config.items()])}"})

if __name__ == "__main__":
    games = [l for l in open("raw/raw.uci").read().splitlines() if l.strip()]
    bchsz = 512
    N = len(games) // bchsz
    batches = [games[i*bchsz:(i*bchsz)+bchsz] for i in range(N)]
    with pq.ParquetWriter("data.parquet", SCHEMA, compression="snappy") as writer:
        with Pool(initializer=init_worker) as pool:
            for hs, zs_batch, ms_batch in tqdm(pool.imap_unordered(process_batch, batches, chunksize=1), total=len(batches), unit="batch"):
                writer.write_table(pa.table({"fen_position": hs, "uci_move_played": ms_batch, "stockfish_wdl": zs_batch}, schema=SCHEMA))
