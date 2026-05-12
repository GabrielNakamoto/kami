from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
import numpy as np, wandb
from model import Model
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters, safe_save, get_state_dict
from tinygrad.dtype import dtypes
from tinygrad.device import Device

print(Device.DEFAULT)

N = 10000000
X_pieces = np.lib.format.open_memmap("pieces.np", shape=(N,512))
X_games = np.lib.format.open_memmap("globals.np", shape=(N,7))
Y_move = np.lib.format.open_memmap("moves.np", shape=(N,4672))
Y_outcome = np.lib.format.open_memmap("outcomes.np", shape=(N,))

d_model = 256
depth = 10
heads = 8
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
        "n_params" : n_params,
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

eval_xp, eval_xg, eval_yp, eval_yv = random_batch(1024)

@TinyJit
def step(xp, xg, yp, yv):
    optim.zero_grad()
    policy_logits, value_logits = model(xp, xg)
    policy_logits = policy_logits.masked_fill(yp < 0, -1e9)
    yp = yp.maximum(0)
    policy_loss = policy_logits.cross_entropy(yp)
    value_loss = (yv.squeeze(-1) - value_logits).square().mean()
    loss = policy_loss + c_value * value_loss
    loss.backward()
    optim.step()
    return loss, policy_loss, value_loss

for t in range(steps):
    if t < warmup_steps: 
        lr = peak_lr * (t + 1) / warmup_steps
    else:
        progress = (t - warmup_steps) / (steps - warmup_steps)
        lr = peak_lr * 0.5 * (1 + np.cos(np.pi * progress))
    optim.lr.assign(Tensor([lr]))

    Tensor.training = True
    loss, policy_loss, value_loss = step(*random_batch(batch_size))
    Tensor.training = False
    
    if t % 100 == 0:
        preds = model(eval_xp, eval_xg)[0].masked_fill(eval_yp < 0, -1e9).argmax(axis=-1)
        targets = eval_yp.maximum(0).argmax(axis=-1)
        acc = (preds == targets).float().mean().item()
        # print(f"step: {t}, loss={loss.item():.2f}, acc={acc*100.:.2f}%")
        run.log({"acc":acc*100, "loss":loss.item(), "policy_loss" : policy_loss.item(), "value_loss" : value_loss.item()})
    if t % 1000:
        safe_save(get_state_dict(model), "model.safetensors", metadata=run.config.as_dict())
