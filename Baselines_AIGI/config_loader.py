"""
Configuration loader for AIGC Detection project.
Loads configuration from YAML files and provides easy access to parameters.
"""

import yaml
import os
import argparse
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """
    Configuration loader that handles YAML configuration files.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = {}
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dictionary containing the configuration
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        return self.config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Key in dot notation (e.g., 'data.train_dataroot')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any):
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Key in dot notation (e.g., 'data.train_dataroot')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
    
    def update_from_args(self, args: argparse.Namespace):
        """
        Update configuration with command line arguments.
        
        Args:
            args: Parsed command line arguments
        """
        # Update paths if provided
        if hasattr(args, 'train_dataroot') and args.train_dataroot:
            self.set('data.train_dataroot', args.train_dataroot)
        if hasattr(args, 'test_dataroot') and args.test_dataroot:
            self.set('data.test_dataroot', args.test_dataroot)
        if hasattr(args, 'checkpoints_dir') and args.checkpoints_dir:
            self.set('models.checkpoints_dir', args.checkpoints_dir)
        
        # Update training parameters if provided
        if hasattr(args, 'lr') and args.lr:
            self.set('training.lr', args.lr)
        if hasattr(args, 'batch_size') and args.batch_size:
            self.set('training.batch_size', args.batch_size)
        if hasattr(args, 'niter') and args.niter:
            self.set('training.niter', args.niter)
        if hasattr(args, 'gpu_ids') and args.gpu_ids:
            self.set('training.gpu_ids', args.gpu_ids)
        if hasattr(args, 'name') and args.name:
            self.set('training.name', args.name)
    
    def create_directories(self):
        """
        Create necessary directories based on configuration.
        """
        # Create output directories
        dirs_to_create = [
            self.get('models.checkpoints_dir'),
            self.get('models.output.hf_model_output'),
            self.get('models.output.visual_encoder_output'),
            self.get('models.output.llava_output'),
        ]
        
        for dir_path in dirs_to_create:
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
    
    def to_namespace(self) -> argparse.Namespace:
        """
        Convert configuration to argparse Namespace for compatibility.
        
        Returns:
            Namespace containing all configuration parameters
        """
        namespace = argparse.Namespace()
        
        # Flatten configuration to namespace
        def flatten_config(config, parent_key=''):
            items = []
            for k, v in config.items():
                new_key = f"{parent_key}_{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_config(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        flat_config = flatten_config(self.config)
        
        # Set common parameters for compatibility with existing code
        namespace.dataroot = self.get('data.train_dataroot')
        namespace.test_dataroot = self.get('data.test_dataroot')
        namespace.train_split = self.get('data.train_split', 'train')
        namespace.val_split = self.get('data.val_split', 'val')
        namespace.checkpoints_dir = self.get('models.checkpoints_dir')
        namespace.arch = self.get('training.arch', 'res50')
        namespace.trainmode = self.get('training.trainmode', 'lora')
        namespace.modelname = self.get('training.modelname', 'CLIP:ViT-L/14@336px')
        namespace.lr = self.get('training.lr', 0.0001)
        namespace.batch_size = self.get('training.batch_size', 16)
        namespace.niter = self.get('training.niter', 1000)
        namespace.delr_freq = self.get('training.delr_freq', 20)
        namespace.loss_freq = self.get('training.loss_freq', 400)
        namespace.data_aug = self.get('training.data_aug', True)
        namespace.loadSize = self.get('training.loadSize', 384)
        namespace.cropSize = self.get('training.cropSize', 336)
        namespace.gpu_ids = self.get('training.gpu_ids', '0')
        namespace.name = self.get('training.name', 'AIGC_Detection')
        namespace.num_threads = self.get('system.num_threads', 8)
        namespace.classes = ""
        namespace.isTrain = True
        
        # Add all flattened config items
        for k, v in flat_config.items():
            setattr(namespace, k, v)
        
        return namespace
    
    def save_config(self, output_path: str):
        """
        Save current configuration to a YAML file.
        
        Args:
            output_path: Path to save the configuration file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    def print_config(self):
        """
        Print the current configuration in a formatted way.
        """
        print("=" * 80)
        print("Configuration:")
        print("=" * 80)
        
        def print_dict(d, indent=0):
            for k, v in d.items():
                if isinstance(v, dict):
                    print("  " * indent + f"{k}:")
                    print_dict(v, indent + 1)
                else:
                    print("  " * indent + f"{k}: {v}")
        
        print_dict(self.config)
        print("=" * 80)


def create_argument_parser():
    """
    Create argument parser for command line interface.
    
    Returns:
        ArgumentParser object
    """
    parser = argparse.ArgumentParser(
        description="AIGC Detection Training with Configuration File Support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration file
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Path to configuration file'
    )
    
    # Override options
    parser.add_argument(
        '--train_dataroot', 
        type=str, 
        help='Training data directory (overrides config)'
    )
    parser.add_argument(
        '--test_dataroot', 
        type=str, 
        help='Test data directory (overrides config)'
    )
    parser.add_argument(
        '--checkpoints_dir', 
        type=str, 
        help='Checkpoints directory (overrides config)'
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--niter', 
        type=int, 
        help='Number of iterations (overrides config)'
    )
    parser.add_argument(
        '--gpu_ids', 
        type=str, 
        help='GPU IDs (overrides config)'
    )
    parser.add_argument(
        '--name', 
        type=str, 
        help='Experiment name (overrides config)'
    )
    
    return parser


def load_config_from_args(args: argparse.Namespace) -> ConfigLoader:
    """
    Load configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        ConfigLoader instance
    """
    config_loader = ConfigLoader(args.config)
    config_loader.update_from_args(args)
    config_loader.create_directories()
    
    return config_loader


if __name__ == "__main__":
    # Test the configuration loader
    parser = create_argument_parser()
    args = parser.parse_args()
    
    config_loader = load_config_from_args(args)
    config_loader.print_config() 