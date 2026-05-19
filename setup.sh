#!/bin/bash
apt-get update && apt-get -y install clang
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
wandb login
hf auth login
python3 data/process.py
python3 data/tables.py
WANDB=1 python3 train.py
