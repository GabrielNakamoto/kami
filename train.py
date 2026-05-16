from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
import numpy as np, wandb, os
from model import Model
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters, safe_save, get_state_dict
from tinygrad.dtype import dtypes
from tinygrad.device import Device
from datasets import load_dataset_builder

print(Device.DEFAULT)

N = int(load_dataset_builder("gRa1ne/decorrelated-chess-3.8m").info.splits['train'].num_examples)
x = Tensor.empty(N, 64, dtype=dtypes.uint8, device="DISK:data/tensors/x.bin").to('CPU')
xi = Tensor.empty(N, 9, dtype=dtypes.uint8, device="DISK:data/tensors/xi.bin").to('CPU')
yp = Tensor.empty(N, 1858, dtype=dtypes.int8, device="DISK:data/tensors/yp.bin").to('CPU')
yz = Tensor.empty(N, 3, dtype=dtypes.float32, device="DISK:data/tensors/yz.bin").to('CPU')

valid_N = 1000
train_N = N - valid_N

def random_batch(n):
    samples = Tensor.randint(n, high=train_N, dtype=dtypes.uint32)
    return x[samples].to(Device.DEFAULT), xi[samples].to(Device.DEFAULT).float(), yp[samples].to(Device.DEFAULT).float()

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

def eval_model():
    exp = x[-valid_N:].to(Device.DEFAULT)
    exi = xi[-valid_N:].to(Device.DEFAULT).float()
    eyp = yp[-valid_N:].to(Device.DEFAULT).float()
    preds = model(exp, exi).masked_fill(eyp < 0, -1e9).argmax(axis=-1)
    targets = eyp.maximum(0).argmax(axis=-1)
    return (preds == targets).float().mean().item()

@TinyJit
def step(xp, xg, yp):
    optim.zero_grad()
    policy_logits = model(xp, xg)
    policy_logits = policy_logits.masked_fill(yp < 0, -1e9)
    yp = yp.maximum(0)
    loss = policy_logits.cross_entropy(yp).backward()
    # policy_loss = -(policy_logits.log_softmax(-1) * yp).sum(-1).mean()
    # value_loss = (yv.squeeze(-1) * 0.9 - value_logits).square().mean()
    # loss = policy_loss + c_value * value_loss
    # loss.backward()
    optim.step()
    return loss

for t in range(steps):
    Tensor.training = True
    if t < warmup_steps: 
        lr = peak_lr * (t + 1) / warmup_steps
    else:
        progress = (t - warmup_steps) / (steps - warmup_steps)
        lr = peak_lr * 0.5 * (1 + np.cos(np.pi * progress))
    optim.lr.assign(Tensor([lr]))

    loss = step(*random_batch(batch_size))
    
    if t % 10 == 0:
        Tensor.training = False
        acc = eval_model()
        if run: 
            run.log({"acc":acc*100, "loss":loss.item()})
        else:
            print(f"step: {t}, loss={loss.item():.2f}, acc={acc*100.:.2f}%")
    if t % 1000 == 0:
        Tensor.training = False
        safe_save(get_state_dict(model), "model.safetensors", metadata=config)
