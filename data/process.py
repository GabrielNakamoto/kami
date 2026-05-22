import chess, numpy as np, os
from multiprocessing import Pool
from tqdm import tqdm
from datasets import load_dataset
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from util.convert import board_to_tensor, uci_move_to_tensor, get_global_features

def tensorize_batch(batch):
    xs, xis, yzs, yps = [], [], [], []
    for fen, move, z in zip(batch['fens'], batch['move_played'], batch['wdl']):
        player = fen.split()[-5] == 'w'
        board = chess.Board(fen)
        xis.append(get_global_features(board, player))
        xs.append(board_to_tensor(board, not player))
        yzs.append(np.eye(3, dtype=np.float32)[z])
        yps.append(uci_move_to_tensor(fen, move, player))
    return xs, xis, yzs, yps

if __name__ == "__main__":
    OUT_DIR = "tensors"
    os.makedirs(OUT_DIR, exist_ok=True)
    dataset = load_dataset("gRa1ne/decorrelated-chess", split='train')
    N = len(dataset)
    batches = dataset.batch(256)

    x = np.memmap(f"{OUT_DIR}/x.bin", dtype=np.uint8, mode="w+", shape=(N, 64))
    xi = np.memmap(f"{OUT_DIR}/xi.bin", dtype=np.uint8, mode="w+", shape=(N, 9))
    yz = np.memmap(f"{OUT_DIR}/yz.bin", dtype=np.uint8, mode="w+", shape=(N, 3))
    yp = np.memmap(f"{OUT_DIR}/yp.bin", dtype=np.int8, mode="w+", shape=(N, 1858))

    start = 0
    with Pool() as pool:
        for xs, xis, yzs, yps in tqdm(pool.imap_unordered(tensorize_batch, batches, chunksize=1), total=N//256, unit="batch"):
            end = min(start+256,N)
            x[start:end]=np.stack(xs).astype(np.uint8)
            xi[start:end]=np.stack(xis).astype(np.uint8)
            yz[start:end]=np.stack(yzs)
            yp[start:end]=np.stack(yps).astype(np.int8)
            start += 256

    move_map = Tensor.empty(1858-66, device="DISK:tensors/move_map.bin", dtype=dtypes.int32)
    non_promos = [fr*64 + to_ for fr, to_, p in build_move_mapping().keys() if not p]
    move_map.assign(Tensor(sorted(non_promos)))
