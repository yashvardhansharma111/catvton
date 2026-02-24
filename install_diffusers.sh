#!/bin/bash
# Run this if diffusers is missing after pip install -r requirements.txt
# (git install can fail to complete; this installs diffusers explicitly.)

set -e
echo "Using: $(which python) $(which pip)"
echo "Installing diffusers from GitHub..."
pip install --no-cache-dir "git+https://github.com/huggingface/diffusers.git"
echo "Checking..."
python -c "from diffusers.image_processor import VaeImageProcessor; print('OK: diffusers installed')"
