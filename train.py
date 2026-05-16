from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
import numpy as np, wandb, os
from model import Model
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters, safe_save, get_state_dict
from tinygrad.dtype import dtypes
from tinygrad.device import Device

print(Device.DEFAULT)

N = 100000
x = Tensor.empty(N, 8, 64, dtype=dtypes.uint8, device="DISK:data/tensors/x.bin")
yp = Tensor.empty(N, 1858, dtype=dtypes.int8, device="DISK:data/tensors/yp.bin")
yz = Tensor.empty(N, 3, dtype=dtypes.float32, device="DISK:data/tensors/yz.bin")

valid_N = 5000
train_N = N - valid_N

def random_batch(n):
    samples = np.random.randint(0, train_N, size=n)
    xp = Tensor(np.asarray(X_pieces[samples]), dtype=dtypes.float32)
    xg = Tensor(np.asarray(X_games[samples]), dtype=dtypes.float32)
    Y_p = Tensor(np.asarray(Y_move[samples]), dtype=dtypes.float32)
    Y_v = Tensor(np.asarray(Y_outcome[samples]), dtype=dtypes.float32)
    return xp, xg, Y_p, Y_v

d_model = 128
depth = 6
heads = 4
c_value = 0.1
batch_size = 8
warmup_steps = 2e3
peak_lr = 4e-4
steps = 100000

model = Model(d_model, depth, heads, use_lc_attn=True)
optim = AdamW(get_parameters(model), lr=1e-3, weight_decay=0.01)

n_params = sum(map(Tensor.numel, get_state_dict(model).values()))

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

run = wandb.init(entity="raine1-me", project="chessformer", config=config) if os.getenv('WANDB', False) else None


eval_xp = Tensor(np.asarray(X_pieces[-valid_N:]), dtype=dtypes.float32)
eval_xg = Tensor(np.asarray(X_games[-valid_N:]), dtype=dtypes.float32)
eval_yp = Tensor(np.asarray(Y_move[-valid_N:]), dtype=dtypes.float32)
eval_yv = Tensor(np.asarray(Y_outcome[-valid_N:]), dtype=dtypes.float32)

@TinyJit
def step(xp, xg, yp, yv):
    optim.zero_grad()
    policy_logits, value_logits = model(xp, xg)
    policy_logits = policy_logits.masked_fill(yp < 0, -1e9)
    yp = yp.maximum(0)
    policy_loss = -(policy_logits.log_softmax(-1) * yp).sum(-1).mean()
    # policy_loss = policy_logits.cross_entropy(yp)
    value_loss = (yv.squeeze(-1) * 0.9 - value_logits).square().mean()
    loss = policy_loss + c_value * value_loss
    loss.backward()
    optim.step()
    return loss, policy_loss, value_loss

for t in range(steps):
    Tensor.training = True
    if t < warmup_steps: 
        lr = peak_lr * (t + 1) / warmup_steps
    else:
        progress = (t - warmup_steps) / (steps - warmup_steps)
        lr = peak_lr * 0.5 * (1 + np.cos(np.pi * progress))
    optim.lr.assign(Tensor([lr]))

    loss, policy_loss, value_loss = step(*random_batch(batch_size))
    
    if t % 10 == 0:
        Tensor.training = False
        preds = model(eval_xp, eval_xg)[0].masked_fill(eval_yp < 0, -1e9).argmax(axis=-1)
        targets = eval_yp.maximum(0).argmax(axis=-1)
        acc = (preds == targets).float().mean().item()
        if run: run.log({"acc":acc*100, "loss":loss.item(), "policy_loss" : policy_loss.item(), "value_loss" : value_loss.item()})
        else: print(f"step: {t}, loss={loss.item():.2f}, policy={policy_loss.item():.2f}, value={value_loss.item():.2f}, acc={acc*100.:.2f}%")
    if t % 1000 == 0:
        Tensor.training = False
        safe_save(get_state_dict(model), "model.safetensors", metadata=config)
