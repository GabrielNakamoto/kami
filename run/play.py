import os, sys, struct, json, chess, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from flask import Flask, request, jsonify, send_from_directory
from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_load, load_state_dict
from model import Model
from data.process import board_to_tensor, move_to_idx

WEIGHTS = "model.safetensors"
metadata = {}
if os.path.exists(WEIGHTS):
    with open(WEIGHTS, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        metadata = json.loads(f.read(n)).get("__metadata__", {})

dim = int(metadata.get("hidden", 128))
depth = int(metadata.get("depth", 5))
heads = int(metadata.get("heads", 4))
model = Model(dim, depth, heads, use_lc_attn=True)
if os.path.exists(WEIGHTS):
    load_state_dict(model, safe_load(WEIGHTS))
    print(f"loaded {WEIGHTS}")
else:
    print(f"WARNING: {WEIGHTS} not found, using random weights")
Tensor.training = False

def features(board: chess.Board, player: bool) -> np.ndarray:
    return np.array([
        board.has_kingside_castling_rights(player),
        board.has_queenside_castling_rights(player),
        board.has_kingside_castling_rights(not player),
        board.has_queenside_castling_rights(not player),
        board.ep_square if board.ep_square else 0,
        board.has_legal_en_passant(),
        board.halfmove_clock,
        board.is_repetition(2),
        board.is_repetition(3),
    ], dtype=np.float32)

def legal_move_dist(uci_moves: list[str]):
    board = chess.Board()
    for uci in uci_moves: board.push_uci(uci)
    if board.is_game_over(): return None, [], True
    player = board.turn == chess.WHITE
    flip = not player
    xp = Tensor(board_to_tensor(board, flip).astype(np.uint8)).unsqueeze(0)
    xg = Tensor(features(board, player)).unsqueeze(0)
    logits = model(xp, xg).numpy()[0]  # (1858,)
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

app = Flask(__name__)

@app.route("/")
def index(): return send_from_directory(".", "index.html")

@app.route("/metadata")
def meta(): return jsonify(metadata)

@app.route("/move", methods=["POST"])
def move():
    mv, dist, over = legal_move_dist(request.json.get("moves", []))
    if over: return jsonify({"move": None, "over": True, "dist": []})
    return jsonify({"move": mv.uci(), "dist": dist})

if __name__ == "__main__":
    app.run(port=5000, debug=False)
