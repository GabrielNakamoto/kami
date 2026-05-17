from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
import numpy as np, wandb, os
from model import Model
from tinygrad.nn.optim import AdamW, Muon
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

valid_N = 5000
train_N = N - valid_N

config={
    "hidden" : 128,
    "depth" : 5,
    "heads" : 4,
    # "n_params" : n_params,
    "c_value" : 0.0,
    "batch_size" : 512,
    "training_steps" : 50000,
    "training_examples" : train_N
    # "peak_lr" : peak_lr,
}

def random_batch():
    samples = Tensor.randint(config['batch_size'], high=train_N, dtype=dtypes.uint32)
    return x[samples].to(Device.DEFAULT), xi[samples].to(Device.DEFAULT).float(), yp[samples].to(Device.DEFAULT).float()

model = Model(config['hidden'], config['depth'], config['heads'], use_lc_attn=True)
config['n_params'] = sum(map(Tensor.numel, get_state_dict(model).values()))
params = get_parameters(model)
matrix_params = [p for p in params if p.ndim == 2]
highdim_params = [p for p in params if p.ndim != 2]

opt1 = Muon(matrix_params)
opt2 = AdamW(highdim_params)

logger = wandb.init(entity="raine1-me", project="chessformer", config=config) if os.getenv('WANDB', False) else None

def eval_model():
    exp = x[-valid_N:].to(Device.DEFAULT)
    exi = xi[-valid_N:].to(Device.DEFAULT).float()
    eyp = yp[-valid_N:].to(Device.DEFAULT).float()
    preds = model(exp, exi).masked_fill(eyp < 0, -1e9).argmax(axis=-1)
    targets = eyp.maximum(0).argmax(axis=-1)
    return (preds == targets).float().mean().item()

@TinyJit
def step(xp, xg, yp):
    opt1.zero_grad()
    opt2.zero_grad()
    policy_logits = model(xp, xg)
    policy_logits = policy_logits.masked_fill(yp < 0, -1e9)
    yp = yp.maximum(0)
    loss = policy_logits.cross_entropy(yp).backward()
    opt1.step()
    opt2.step()
    return loss

for t in range(config['training_steps']):
    Tensor.training = True
    loss = step(*random_batch())
    
    if t % 10 == 0:
        Tensor.training = False
        acc = eval_model()
        if logger: logger.log({"acc":acc*100, "loss":loss.item()})
        else: print(f"step: {t}, loss={loss.item():.2f}, acc={acc*100.:.2f}%")
    if t % 1000 == 0:
        Tensor.training = False
        safe_save(get_state_dict(model), "model.safetensors", metadata=config)
