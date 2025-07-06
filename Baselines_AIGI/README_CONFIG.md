# AIGC Detection Configuration System User Guide

This document explains how to use the configuration system for AIGC detection model training and conversion.

## Overview

The configuration system extracts all hardcoded paths and parameters into YAML configuration files, supporting:
- Visual pretraining
- Model conversion to HuggingFace format
- LLaVA model vision component replacement
- Complete training and conversion pipeline

## Main Files

- `config.yaml`: Main configuration file
- `config_loader.py`: Configuration loader
- `train_with_config.py`: Unified training script
- `convert_weight2hf_config.py`: Model conversion script
- `transform_vision_model_config.py`: Vision model replacement script

## Configuration File Structure

```yaml
# Data paths
data:
  train_dataroot: "/path/to/training/data"
  test_dataroot: "/path/to/test/data"
  
# Model paths
models:
  checkpoints_dir: "./checkpoints"
  pretrained:
    clip_model: "/path/to/clip/model"
    llava_model: "/path/to/llava/model"
  output:
    hf_model_output: "./output/hf_models"
    visual_encoder_output: "./output/visual_encoder"
    llava_output: "./output/llava_modified"

# Training parameters
training:
  trainmode: "lora"
  modelname: "CLIP:ViT-L/14@336px"
  lr: 0.0001
  batch_size: 16
  niter: 1000
  # ... other parameters
```

## Usage

### 1. Prepare Configuration File

First copy and modify the `config.yaml` file with correct paths:

```bash
cp config.yaml my_config.yaml
# Edit my_config.yaml to set correct paths
```

### 2. Complete Training Pipeline

Use the unified training script to run the complete training and conversion pipeline:

```bash
python train_with_config.py --config my_config.yaml
```

This will execute the following steps:
1. Visual pretraining
2. Model conversion to HuggingFace format
3. LLaVA model vision component replacement

### 3. Run Individual Components

#### Model conversion only

```bash
python convert_weight2hf_config.py --config my_config.yaml
```

#### Vision model replacement only

```bash
python transform_vision_model_config.py --config my_config.yaml
```

#### Test model performance

```bash
python convert_weight2hf_config.py --config my_config.yaml --test_only
```

### 4. Command Line Parameter Override

You can override configuration file settings with command line arguments:

```bash
python train_with_config.py --config my_config.yaml \
    --train_dataroot /new/path/to/training/data \
    --batch_size 32 \
    --lr 0.0002
```

## Configuration Parameters Detailed

### Data Configuration (data)

- `train_dataroot`: Training data root directory
- `test_dataroot`: Test data root directory
- `train_split`: Training set directory name (default: "train")
- `val_split`: Validation set directory name (default: "val")

### Model Configuration (models)

- `checkpoints_dir`: Model checkpoint save directory
- `pretrained.clip_model`: Pretrained CLIP model path
- `pretrained.llava_model`: Pretrained LLaVA model path
- `output.hf_model_output`: HuggingFace model output directory
- `output.visual_encoder_output`: Visual encoder output directory
- `output.llava_output`: Modified LLaVA model output directory

### Training Configuration (training)

- `trainmode`: Training mode ("NPR", "lora", "CNNDetection", or "rine")
- `modelname`: Model name
- `lr`: Learning rate
- `batch_size`: Batch size
- `niter`: Number of training epochs
- `delr_freq`: Learning rate decay frequency
- `data_aug`: Whether to enable data augmentation
- `loadSize`: Image loading size
- `cropSize`: Image cropping size

### Testing Configuration (testing)

- `test_vals`: List of test dataset names
- `multiclass`: Corresponding multiclass flags
- `batch_size`: Test batch size
- `noise_type`: Noise type
- `noise_ratio`: Noise ratio

### Visual Pretraining Configuration (visual_pretraining)

- `enabled`: Whether to enable visual pretraining
- `checkpoint_path`: Pretrained checkpoint path
- `auto_convert_to_hf`: Automatically convert to HuggingFace format
- `auto_replace_llava_vision`: Automatically replace LLaVA vision model

## Directory Structure

Directory structure after training completion:

```
./checkpoints_new/
├── AIGC_Detection_YYYY_MM_DD_HH_MM_SS/
│   ├── model_epoch_*.pth
│   ├── training.log
│   └── tensorboard_logs/
./output/
├── hf_models/
├── visual_encoder/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── proj.pth
│   └── fc.pth
└── llava_modified/
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer files
```

## Important Notes

1. **Path Setup**: Ensure all paths are correctly set, especially pretrained model paths
2. **GPU Memory**: Adjust batch_size according to GPU memory
3. **Dependencies**: Ensure all necessary dependencies are installed:
   ```bash
   pip install -r requirements_config.txt
   ```
4. **Data Format**: Ensure training and test data are organized in the correct format

## Troubleshooting

### Common Issues

1. **Configuration file not found**
   ```
   FileNotFoundError: Configuration file not found: config.yaml
   ```
   Solution: Check if the configuration file path is correct

2. **Pretrained model not found**
   ```
   Error: CLIP model not found at: /path/to/clip/model
   ```
   Solution: Download and set the correct pretrained model path

3. **GPU out of memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   Solution: Reduce batch_size or use a smaller model

4. **Permission error**
   ```
   PermissionError: [Errno 13] Permission denied
   ```
   Solution: Check write permissions for output directories

### Debugging Tips

1. Use `--test_only` parameter to test model without conversion
2. Check tensorboard logs to monitor training process
3. View generated log files for detailed information

## Example Configuration

Refer to the example configuration in the `config_example.yaml` file and adjust according to your environment.

## Support

If you encounter issues, please check:
1. Whether dependencies are correctly installed
2. Whether configuration file paths are correct
3. Whether data format meets requirements
4. Whether GPU memory is sufficient 