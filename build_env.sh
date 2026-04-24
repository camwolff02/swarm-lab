#!/usr/bin/env bash
git submodule update --init --recursive --progress
uv sync --extra detatched --reinstall-package cpsquare-lab
uv pip install --upgrade pip
source .venv/bin/activate
cd ../IsaacLab
./isaaclab.sh -i assets,contrib,newton,ov,physx,rl[rsl_rl,skrl],tasks,teleop,visualizers[all]
