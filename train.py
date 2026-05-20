from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
import wandb, os
from model import Model
from tinygrad.nn.optim import AdamW, Muon
from tinygrad.nn.state import get_parameters, safe_save, get_state_dict
from tinygrad.dtype import dtypes
from tinygrad.device import Device
from datasets import load_dataset_builder

N = int(load_dataset_builder("gRa1ne/decorrelated-chess-3.8m").info.splits['train'].num_examples)
x = Tensor.empty(N, 64, dtype=dtypes.uint8, device="DISK:tensors/x.bin").to('CPU')
xi = Tensor.empty(N, 9, dtype=dtypes.uint8, device="DISK:tensors/xi.bin").to('CPU')
yp = Tensor.empty(N, 1858, dtype=dtypes.int8, device="DISK:tensors/yp.bin").to('CPU')
yz = Tensor.empty(N, 3, dtype=dtypes.float32, device="DISK:tensors/yz.bin").to('CPU')

valid_N = 5000
train_N = N - valid_N

config={
    "hidden" : 128,
    "depth" : 5,
    "heads" : 4,
    "c_value" : 0.0,
    "batch_size" : 512,
    "training_steps" : 50000,
    "training_examples" : train_N
}

def random_batch():
    samples = Tensor.randint(config['batch_size'], high=train_N, dtype=dtypes.uint32, device='CPU')
    return x[samples].to(Device.DEFAULT), xi[samples].to(Device.DEFAULT).float(), yp[samples].to(Device.DEFAULT).float()

model = Model(config['hidden'], config['depth'], config['heads'], use_lc_attn=True)
params = get_parameters(model)
config['n_params'] = sum(map(Tensor.numel, params))

matrix_params = [p for p in params if p.ndim == 2]
highdim_params = [p for p in params if p.ndim != 2]

opt1 = Muon(matrix_params)
opt2 = AdamW(highdim_params)

logger = wandb.init(entity="raine1-me", project="chessformer", config=config) if os.getenv('WANDB', False) else None

def eval_model():
    exp, exi, eyp = x[-valid_N:].to(Device.DEFAULT), xi[-valid_N:].to(Device.DEFAULT).float(), yp[-valid_N:].to(Device.DEFAULT).float()
    logits = model(exp, exi).masked_fill(eyp < 0, -1e9)
    preds = logits.argmax(axis=-1)
    eyp = eyp.maximum(0)
    loss = logits.cross_entropy(eyp)
    targets = eyp.argmax(axis=-1)
    return (preds == targets).float().mean().item(), loss

@TinyJit
def step(xp, xg, yp):
    opt1.zero_grad(); opt2.zero_grad()
    policy_logits = model(xp, xg).masked_fill(yp < 0, -1e9)
    yp = yp.maximum(0)
    loss = policy_logits.cross_entropy(yp).backward()
    opt1.step(); opt2.step()
    return loss

print("Device:", Device.DEFAULT)
print(f"Model size: {config['n_params']/1e6:.2f}m params")

for t in range(config['training_steps']):
    Tensor.training = True
    loss = step(*random_batch())
    
    if t % 10 == 0:
        Tensor.training = False
        acc, valid_loss = eval_model()
        print(f"step: {t:5d}, loss={loss.item():.2f}, valid acc={acc*100.:.2f}%, valid loss={valid_loss.item():.2f}")
        if logger: logger.log({"acc":acc*100, "train_loss":loss.item(), "valid_loss":valid_loss.item()})
    if t % 1000 == 0:
        Tensor.training = False
        safe_save(get_state_dict(model), "model.safetensors", metadata=config)
