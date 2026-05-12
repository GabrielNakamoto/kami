#!/bin/bash

python3 -m venv .venv
source .venv/bin/activate
pip install chess numpy tinygrad wandb huggingface_hub datasets
wandb login
python3 data.py 10000000
python3 train.py
