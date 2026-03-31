#!/bin/bash
# Setup script for RT-DETR project

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "Setup complete! To activate the environment, run:"
echo "source venv/bin/activate"
