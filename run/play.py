from math import dist
import os, sys, struct, json, chess, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.nn.state import safe_load, load_state_dict
from model import Model, MCTS, BatchMCTS
from util.convert import board_to_tensor, move_to_idx, get_global_features

WEIGHTS = "model.safetensors"
metadata = {}
if os.path.exists(WEIGHTS):
    with open(WEIGHTS, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        metadata = json.loads(f.read(n)).get("__metadata__", {})

# flatten dict 1 layer deep
flattened_md = {}
for k, v in metadata.items():
    if isinstance(v, dict):
        for kk, vv in v.items(): flattened_md[f"{k}_{kk}"]=vv
    else: flattened_md[k]=v

dim = int(metadata["training"].get("hidden_dimension", 128))
depth = int(metadata["training"].get("transformer_blocks", 5))
heads = int(metadata["training"].get("attention_heads", 4))
model = Model(dim, depth, heads, use_lc_attn=True)
if os.path.exists(WEIGHTS):
    load_state_dict(model, safe_load(WEIGHTS))
    print(f"loaded {WEIGHTS}")
else:
    print(f"WARNING: {WEIGHTS} not found, using random weights")
# search = MCTS(model)
search = BatchMCTS(model)
Tensor.training = False

def legal_move_dist(fen: str):
    board = chess.Board(fen)
    if board.is_game_over(): return None, [], True
    player = board.turn == chess.WHITE
    flip = not player
    xp = Tensor(board_to_tensor(board, flip).astype(np.uint8)).unsqueeze(0)
    xg = Tensor(get_global_features(board, player)).unsqueeze(0)
    logits = model(xp, xg)[0].numpy()[0]  # (1858,)
    moves, scores = [], []
    for mv in board.generate_legal_moves():
        moves.append(mv)
        scores.append(float(logits[move_to_idx(mv, flip)]))
    scores = np.array(scores)
    probs = np.exp(scores - scores.max())
    probs = probs / probs.sum()
    order = np.argsort(-probs)
    dist = [{"move": moves[i].uci(), "logit": scores[i], "prob": float(probs[i])} for i in order]
    return moves[int(order[0])], dist, False

def make_dist(moves, nsa):
    visits = np.asarray(nsa, dtype=np.float64)
    total = visits.sum()
    probs = visits / total if total > 0 else visits
    order = np.argsort(-visits)
    return [{"move": moves[i].uci(), "visits": int(visits[i]), "prob": float(probs[i])}
            for i in order]

app = Flask(__name__)

@app.route("/")
def index(): return send_from_directory(".", "index.html")

@app.route("/metadata")
def meta():
    return jsonify(flattened_md)

@app.route("/move", methods=["POST"])
def move():
    fen = request.json.get("fen", "")
    num_batches, batch_size = 32, 32

    @stream_with_context
    def gen():
        moves, nsa = [], None
        for n, moves, nsa in search.search_iter(fen, num_batches=num_batches, batch_size=batch_size):
            payload = {"batch": n, "num_batches": num_batches, "dist": make_dist(moves, nsa)}
            yield json.dumps(payload) + "\n"
        if nsa is not None:
            best = int(np.asarray(nsa).argmax(-1))
            yield json.dumps({"done": True, "move": moves[best].uci(),
                              "dist": make_dist(moves, nsa)}) + "\n"
        else:
            yield json.dumps({"done": True, "move": None, "dist": []}) + "\n"

    return Response(gen(), mimetype="application/x-ndjson")

if __name__ == "__main__":
    app.run(port=5000, debug=False)
