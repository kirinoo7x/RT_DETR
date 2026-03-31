#!/bin/bash
# Simple script to run training with the correct Python environment

cd "$(dirname "$0")"
source venv/bin/activate
python train_rtdetr.py
