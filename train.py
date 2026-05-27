import argparse, yaml
from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
import wandb, os, numpy as np
from model import Model
from tinygrad.nn.optim import AdamW, Muon
from tinygrad.nn.state import get_parameters, safe_save, get_state_dict
from tinygrad.device import Device
from datasets import load_dataset_builder

def main(args):
    config = yaml.safe_load(args.cfg)

    N = int(load_dataset_builder("gRa1ne/decorrelated-chess").info.splits['train'].num_examples)
    x  = np.memmap("tensors/x.bin",  dtype=np.uint8,  mode='r', shape=(N, 64))
    xi = np.memmap("tensors/xi.bin", dtype=np.uint8,  mode='r', shape=(N, 9))
    yp = np.memmap("tensors/yp.bin", dtype=np.int8,   mode='r', shape=(N, 1858))
    yz = np.memmap("tensors/yz.bin", dtype=np.float16,  mode='r', shape=(N, 3))

    def random_batch():
        def mm_to_tensor(arr, idx) -> Tensor: return Tensor(np.array(arr[idx]))
        samples = np.random.randint(0, N - config['dataset']['validation_examples'], config['training']['batch_size'])
        return mm_to_tensor(x, samples), \
            mm_to_tensor(xi, samples).float(), \
            mm_to_tensor(yp, samples).float(), \
            mm_to_tensor(yz, samples).float()


    model = Model(config['training']['hidden_dimension'], config['training']['transformer_blocks'], config['training']['attention_heads'], use_lc_attn=True)
    params = get_parameters(model)
    config['model_params'] = sum(map(Tensor.numel, params))

    print("Device:", Device.DEFAULT)
    print(f"Model size: {config['model_params']/1e6:.2f}m params")
    print(f"Training examples: {(N-config['dataset']['validation_examples'])/1e6:.2f}m")

    matrix_params = [p for p in params if p.ndim == 2]
    highdim_params = [p for p in params if p.ndim != 2]

    opt1 = Muon(matrix_params, lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    opt2 = AdamW(highdim_params, lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])

    logger = wandb.init(entity="raine1-me", project="chessformer", config=config) if args.wandb else None

    vx  = Tensor(np.array(x[-config['dataset']['validation_examples']:]))
    vxi = Tensor(np.array(xi[-config['dataset']['validation_examples']:])).float()
    vyp = Tensor(np.array(yp[-config['dataset']['validation_examples']:])).float()
    vyz = Tensor(np.array(yz[-config['dataset']['validation_examples']:])).float()

    def eval_model(vx, vxi, vyp, vyz):
        policy_logits, value_logits = model(vx, vxi)
        policy_logits = policy_logits.masked_fill(vyp < 0, -1e9)
        preds = policy_logits.argmax(axis=-1)
        yp = vyp.maximum(0)
        loss = policy_logits.cross_entropy(yp) + config['training']['value_loss_weight'] * value_logits.cross_entropy(vyz)
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
        loss = (policy_loss + config['training']['value_loss_weight'] * value_loss).backward()
        opt1.step(); opt2.step()
        return policy_loss, value_loss


    for t in range(config['training']['training_steps']):
        Tensor.training = True
        pl, vl = step(*random_batch())
        
        if t % config['training']['logging_steps'] == 0:
            Tensor.training = False
            acc, valid_loss = eval_model(vx, vxi, vyp, vyz)
            print(f"step: {t:5d}, policy_loss={pl.item():.2f}, value_loss={vl.item():.2f}, valid acc={acc*100.:.2f}%, valid loss={valid_loss.item():.2f}")
            if logger: logger.log({"acc":acc*100, "policy_loss":pl.item(), "value_loss":vl.item(), "valid_loss":valid_loss.item()})
        if t > 0 and t % config['training']['checkpoint_steps'] == 0:
            Tensor.training = False
            safe_save(get_state_dict(model), f"{args.output}.safetensors", metadata=config)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Kami model supervised training pipeline")
    argparser.add_argument('--cfg', type=argparse.FileType('r'), help='yaml config with training params')
    argparser.add_argument('--output', type=str, help='output filename to save/checkpoint model tensors', default="model")
    argparser.add_argument('--wandb', type=bool, help='enable wandb logging', default=False)
    main(argparser.parse_args())
