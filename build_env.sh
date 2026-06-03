#!/usr/bin/env bash
set -euo pipefail

ISAACLAB_DIR="../IsaacLab"
ISAACLAB_REPO_URL="https://github.com/isaac-sim/IsaacLab.git"
ISAACLAB_BRANCH="release/3.0.0-beta2"
ISAACLAB_PIN="aea439d28a" # 2026-05-28: last known-working commit before SKRL 2.1.0 bump

if [ ! -d "$ISAACLAB_DIR" ]; then
  git clone --branch "$ISAACLAB_BRANCH" "$ISAACLAB_REPO_URL" "$ISAACLAB_DIR"
fi

if [ ! -d "$ISAACLAB_DIR/.git" ]; then
  echo "Expected $ISAACLAB_DIR to be a git repository." >&2
  exit 1
fi

git -C "$ISAACLAB_DIR" fetch origin "$ISAACLAB_BRANCH"
git -C "$ISAACLAB_DIR" checkout --detach "$ISAACLAB_PIN"

# Build or refresh the local project env
if [ ! -d ".venv" ]; then
  uv sync --refresh
else
  uv sync --inexact --reinstall-package cpsquare-lab
fi

source .venv/bin/activate

echo "Using Python:"
which python
python -c 'import sys; print(sys.executable)'

# Install IsaacSim
uv pip install "isaacsim[all,extscache]==6.0.0" \
  --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match \
  --prerelease=allow

# Install IsaacLab
cd "$ISAACLAB_DIR"
./isaaclab.sh -i 'teleop,rl[rsl-rl],rl[skrl],visualizer[kit]'

# Install Viser
uv pip install viser==1.0.16
