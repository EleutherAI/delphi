#!/bin/bash
set -x  # Print each command before execution for debugging

# Load .bashrc to get PATH if necessary
source ~/.bashrc

export HOME=/lustre/home/fdraye
echo "Home set to:" $HOME

# Set WANDB_API_KEY to allow wandb to authenticate
export WANDB_API_KEY="097e21df11c8e16d3452a3e5747add10ec3ed5e0"

# Change to the directory where pyproject.toml is located
cd /lustre/home/fdraye/projects/hdelphi
echo "In directory:" $(pwd)

source .venv/bin/activate

cd delphi
python hsae_test.py