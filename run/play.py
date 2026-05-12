import os, struct, json, chess, numpy as np
from flask import Flask, request, jsonify, send_from_directory
from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_load, load_state_dict
from model.game import get_move_idx
from model import Model, GameState

WEIGHTS = "model.safetensors"
model = Model(128, 6, 7, 4)
metadata = {}
if os.path.exists(WEIGHTS):
    load_state_dict(model, safe_load(WEIGHTS))
    with open(WEIGHTS, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        metadata = json.loads(f.read(n)).get("__metadata__", {})
    print(f"loaded {WEIGHTS}")
else:
    print(f"WARNING: {WEIGHTS} not found, using random weights")
Tensor.training = False

def best_move(uci_moves: list[str]) -> tuple[chess.Move | None, bool]:
    state = GameState(chess.Board())
    for uci in uci_moves:
        state.board.push_uci(uci)
        state.update()
    if state.board.is_game_over(): return None, True
    xp, xg = state.piece_tensor().unsqueeze(0), state.global_tensor().unsqueeze(0)
    logits = model(xp, xg)[0].numpy()[0]  # (4672,)
    best, best_score = None, -np.inf
    for mv in state.board.generate_legal_moves():
        fr, ff, n = get_move_idx(mv)
        s = logits[(fr*8 + ff)*73 + n]
        if s > best_score:
            best_score, best = s, mv
    return best, False

app = Flask(__name__)

@app.route("/")
def index(): return send_from_directory(".", "index.html")

@app.route("/metadata")
def meta(): return jsonify(metadata)

@app.route("/move", methods=["POST"])
def move():
    mv, over = best_move(request.json.get("moves", []))
    if over: return jsonify({"move": None, "over": True})
    return jsonify({"move": mv.uci()})

if __name__ == "__main__":
    app.run(port=5000, debug=False)
