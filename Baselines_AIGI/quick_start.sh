#!/bin/bash

# AIGC Detection Configuration System Quick Start Script
# This script helps you quickly set up and run the training pipeline

set -e

echo "=========================================="
echo "AIGC Detection Configuration System"
echo "Quick Start Setup"
echo "=========================================="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed or not in PATH"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements_config.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p ./checkpoints_new
mkdir -p ./output/hf_models
mkdir -p ./output/visual_encoder
mkdir -p ./output/llava_modified

# Copy and customize config file
if [ ! -f "my_config.yaml" ]; then
    echo "Creating customized config file..."
    cp config.yaml my_config.yaml
    echo "Please edit my_config.yaml to set correct paths for your environment."
    echo "Key paths to update:"
    echo "  - data.train_dataroot: Your training data path"
    echo "  - data.test_dataroot: Your test data path"
    echo "  - models.pretrained.clip_model: Path to CLIP model"
    echo "  - models.pretrained.llava_model: Path to LLaVA model"
else
    echo "Configuration file my_config.yaml already exists."
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit my_config.yaml to set correct paths"
echo "2. Run training with: python train_with_config.py --config my_config.yaml"
echo ""
echo "Available scripts:"
echo "  - train_with_config.py: Full training pipeline"
echo "  - convert_weight2hf_config.py: Model conversion only"
echo "  - transform_vision_model_config.py: Vision model replacement only"
echo ""
echo "For more information, see README_CONFIG.md"
echo ""

# Ask user if they want to edit the config file now
read -p "Do you want to edit the config file now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Try to open with common editors
    if command -v nano &> /dev/null; then
        nano my_config.yaml
    elif command -v vim &> /dev/null; then
        vim my_config.yaml
    elif command -v gedit &> /dev/null; then
        gedit my_config.yaml
    else
        echo "Please manually edit my_config.yaml with your preferred editor"
    fi
fi

echo ""
echo "Quick start setup finished!"
echo "You can now run: python train_with_config.py --config my_config.yaml" 