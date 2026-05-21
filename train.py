from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
import wandb, os
from model import Model
from tinygrad.nn.optim import AdamW, Muon
from tinygrad.nn.state import get_parameters, safe_save, get_state_dict
from tinygrad.dtype import dtypes
from tinygrad.device import Device
from datasets import load_dataset_builder

N = int(load_dataset_builder("gRa1ne/decorrelated-chess").info.splits['train'].num_examples)
x = Tensor.empty(N, 64, dtype=dtypes.uint8, device="DISK:tensors/x.bin")
xi = Tensor.empty(N, 9, dtype=dtypes.uint8, device="DISK:tensors/xi.bin")
yp = Tensor.empty(N, 1858, dtype=dtypes.int8, device="DISK:tensors/yp.bin")
yz = Tensor.empty(N, 3, dtype=dtypes.uint8, device="DISK:tensors/yz.bin")

valid_N = 5000
train_N = N - valid_N

config={
    "hidden" : 128,
    "depth" : 5,
    "heads" : 4,
    "batch_size" : 512,
    "training_steps" : 50000,
    "training_examples" : train_N,
    "c_value" : 0.25
}

def get_from_device(t, samples) -> Tensor:
    # switch to numpy memmap?
    return Tensor.stack(*[t[i].to(Device.DEFAULT) for i in sorted(samples)])

vx = x[-valid_N:].contiguous().to(Device.DEFAULT)
vxi = xi[-valid_N:].to(Device.DEFAULT).float()
vyp = yp[-valid_N:].to(Device.DEFAULT).float()
vyz = yz[-valid_N:].to(Device.DEFAULT).float()

def random_batch():
    samples = Tensor.randint(config['batch_size'], high=train_N, dtype=dtypes.uint32).tolist()
    return get_from_device(x, samples), \
        get_from_device(xi, samples).float(), \
        get_from_device(yp, samples).float(), \
        get_from_device(yz, samples).float()

model = Model(config['hidden'], config['depth'], config['heads'], use_lc_attn=True)
params = get_parameters(model)
config['n_params'] = sum(map(Tensor.numel, params))

matrix_params = [p for p in params if p.ndim == 2]
highdim_params = [p for p in params if p.ndim != 2]

opt1 = Muon(matrix_params)
opt2 = AdamW(highdim_params)

logger = wandb.init(entity="raine1-me", project="chessformer", config=config) if os.getenv('WANDB', False) else None

def eval_model():
    global vx, vxi, vyp, vyz
    policy_logits, value_logits = model(vx, vxi)
    policy_logits = policy_logits.masked_fill(vyp < 0, -1e9)
    preds = policy_logits.argmax(axis=-1)
    yp = vyp.maximum(0)
    loss = policy_logits.cross_entropy(yp) + config["c_value"] * value_logits.cross_entropy(vyz)
    targets = yp.argmax(axis=-1)
    return (preds == targets).float().mean().item(), loss

@TinyJit
def step(xp, xg, yp, yz):
    opt1.zero_grad(); opt2.zero_grad()
    policy_logits, value_logits = model(xp, xg)
    policy_logits = policy_logits.masked_fill(yp < 0, -1e9)
    yp = yp.maximum(0)
    policy_loss = policy_logits.cross_entropy(yp)
    value_loss = value_logits.cross_entropy(yz)
    loss = (policy_loss + config["c_value"] * value_loss).backward()
    opt1.step(); opt2.step()
    return loss

print("Device:", Device.DEFAULT)
print(f"Training examples: {N/1e6:.2f}m positions")
print(f"Model size: {config['n_params']/1e6:.2f}m params")

for t in range(config['training_steps']):
    Tensor.training = True
    loss = step(*random_batch())
    
    if t % 10 == 0:
        Tensor.training = False
        acc, valid_loss = eval_model()
        print(f"step: {t:5d}, loss={loss.item():.2f}, valid acc={acc*100.:.2f}%, valid loss={valid_loss.item():.2f}")
        if logger: logger.log({"acc":acc*100, "train_loss":loss.item(), "valid_loss":valid_loss.item()})
    if t > 0 and t % 1000 == 0:
        Tensor.training = False
        safe_save(get_state_dict(model), "model.safetensors", metadata=config)
