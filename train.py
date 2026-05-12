from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
import numpy as np, wandb
from model import Model
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters, safe_save, get_state_dict
from tinygrad.dtype import dtypes
from tinygrad.device import Device

print(Device.DEFAULT)

N = 100000
X_pieces = np.lib.format.open_memmap("pieces.np", shape=(N,512))
X_games = np.lib.format.open_memmap("globals.np", shape=(N,7))
Y_move = np.lib.format.open_memmap("moves.np", shape=(N,4672))
Y_outcome = np.lib.format.open_memmap("outcomes.np", shape=(N,))

d_model = 128
depth = 8
heads = 4
c_value = 0.25
batch_size = 512
warmup_steps = 2e3
peak_lr = 3e-4
steps = 100000

model = Model(d_model, depth, 7, heads)
optim = AdamW(get_parameters(model), lr=3e-4, weight_decay=0.01)

n_params = sum(map(Tensor.numel, get_state_dict(model).values()))

run = wandb.init(
    entity="raine1-me",
    project="chessformer",
    config={
        "d_model" : d_model,
        "depth" : depth,
        "heads" : heads,
        "c_value" : c_value,
        "batch_size" : batch_size,
        "peak_lr" : peak_lr,
        "training_steps" : steps
    }
)

def random_batch(n):
    samples = np.random.randint(0, len(X_pieces), size=n)
    xp = Tensor(np.asarray(X_pieces[samples]), dtype=dtypes.int16)
    xg = Tensor(np.asarray(X_games[samples]), dtype=dtypes.int16)
    Y_p = Tensor(np.asarray(Y_move[samples]), dtype=dtypes.float32)
    Y_v = Tensor(np.asarray(Y_outcome[samples]), dtype=dtypes.float32)
    return xp, xg, Y_p, Y_v

@TinyJit
def step(xp, xg, yp, yv):
    optim.zero_grad()
    policy_logits, value_logits = model(xp, xg)
    policy_logits = policy_logits.masked_fill(yp < 0, -1e9)
    yp = yp.maximum(0)
    loss = policy_logits.cross_entropy(yp) + c_value * (yv.squeeze(-1) - value_logits).square().mean()
    loss.backward()
    optim.step()
    return loss

for t in range(steps):
    if t < warmup_steps: optim.lr.assign(Tensor([peak_lr * (t + 1) / warmup_steps]))
    elif t == warmup_steps: optim.lr.assign(Tensor([peak_lr]))

    Tensor.training = True
    loss = step(*random_batch(batch_size))
    Tensor.training = False
    
    if t % 5 == 0:
        xp, xg, yp, yv = random_batch(256)
        preds = model(xp, xg)[0].masked_fill(yp < 0, -1e9).argmax(axis=-1)
        targets = yp.maximum(0).argmax(axis=-1)
        acc = (preds == targets).float().mean().item()
        print(f"step: {t}, loss={loss.item():.2f}, acc={acc*100.:.2f}%")
        run.log({"acc":acc*100, "loss":loss.item()})
        safe_save(get_state_dict(model), "model.safetensors", metadata=run.config.as_dict())
