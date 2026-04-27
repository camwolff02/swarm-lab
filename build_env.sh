#!/usr/bin/env bash
git submodule update --init --recursive --progress
uv sync --extra detatched --reinstall-package cpsquare-lab
source .venv/bin/activate
cd ../IsaacLab
./isaaclab.sh -i assets,contrib,newton,ov,physx,rl[skrl],tasks,visualizers[all]
