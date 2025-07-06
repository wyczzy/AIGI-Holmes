"""
Enhanced convert_weight2hf.py with configuration file support.
Converts trained models to HuggingFace format for easy deployment and sharing.
"""

import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter
import numpy as np
from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger
from config_loader import ConfigLoader
from transformers import CLIPVisionModel, CLIPImageProcessor

import random


def seed_torch(seed=1029):
    """Set random seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


class WeightConverter:
    """
    Handles the conversion of trained models to HuggingFace format.
    """
    
    def __init__(self, config_loader: ConfigLoader):
        """
        Initialize the weight converter.
        
        Args:
            config_loader: ConfigLoader instance containing all configuration
        """
        self.config = config_loader
        self.setup_environment()
        self.setup_model()
        
    def setup_environment(self):
        """Setup conversion environment."""
        seed_torch(self.config.get('training.seed', 100))
        
        # Create output directories
        self.config.create_directories()
        
    def setup_model(self):
        """Setup the trained model for conversion."""
        # Convert config to namespace for compatibility
        self.opt = self.config.to_namespace()
        
        # Update specific paths
        self.opt.dataroot = self.config.get('data.train_dataroot')
        self.test_dataroot = self.config.get('data.test_dataroot')
        
        # Create model
        self.model = Trainer(self.opt)
        
        # Load the trained checkpoint
        self.load_checkpoint()
        
    def load_checkpoint(self):
        """Load the trained checkpoint."""
        checkpoint_path = self.config.get('visual_pretraining.checkpoint_path')
        
        if not checkpoint_path:
            print("Warning: No checkpoint path specified in configuration")
            return
            
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            return
            
        try:
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # Load different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'fc' in checkpoint:
                    self.model.model.load_state_dict(checkpoint['fc'], strict=False)
                if 'lora' in checkpoint:
                    self.model.model.load_state_dict(checkpoint['lora'], strict=False)
            else:
                # Direct state dict
                self.model.model.load_state_dict(checkpoint, strict=False)
                
            print("Checkpoint loaded successfully")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            
    def test_model(self):
        """Test the loaded model to verify it's working correctly."""
        print("Testing loaded model...")
        
        test_opt = TestOptions().parse(print_options=False)
        test_opt.trainmode = self.config.get('training.trainmode', 'lora')
        test_opt.modelname = self.config.get('training.modelname', 'CLIP:ViT-L/14@336px')
        test_opt.loadSize = self.config.get('training.loadSize', 384)
        test_opt.cropSize = self.config.get('training.cropSize', 336)
        test_opt.batch_size = self.config.get('testing.batch_size', 64)
        test_opt.no_resize = self.config.get('testing.no_resize', False)
        test_opt.no_crop = self.config.get('testing.no_crop', False)
        test_opt.noise_type = self.config.get('testing.noise_type')
        test_opt.noise_ratio = self.config.get('testing.noise_ratio')
        
        test_vals = self.config.get('testing.test_vals', ['test'])
        multiclass = self.config.get('testing.multiclass', [0])
        
        accs = []
        aps = []
        
        for v_id, val in enumerate(test_vals):
            test_opt.dataroot = f'{self.test_dataroot}/{val}'
            test_opt.classes = os.listdir(test_opt.dataroot) if (v_id < len(multiclass) and multiclass[v_id]) else ['']
            
            try:
                acc, ap, r_acc, f_acc, _, _ = validate(self.model.model, test_opt)
                accs.append(acc)
                aps.append(ap)
                
                print(f"({v_id} {val:>10}) acc: {acc*100:.1f}; ap: {ap*100:.1f}, racc: {r_acc*100:.1f}, facc: {f_acc*100:.1f};")
                
            except Exception as e:
                print(f"Error testing on {val}: {e}")
                
        if accs:
            print(f"Mean accuracy: {np.mean(accs)*100:.1f}%, Mean AP: {np.mean(aps)*100:.1f}%")
            
        return np.mean(accs) if accs else 0, np.mean(aps) if aps else 0
        
    def convert_to_huggingface(self):
        """Convert the trained model to HuggingFace format."""
        print("Converting model to HuggingFace format...")
        
        # Get paths
        clip_model_path = self.config.get('models.pretrained.clip_model')
        output_path = self.config.get('models.output.visual_encoder_output')
        
        if not self._validate_conversion_paths(clip_model_path, output_path):
            return False
            
        try:
            # Load base CLIP model
            print(f"Loading base CLIP model from: {clip_model_path}")
            vit = CLIPVisionModel.from_pretrained(clip_model_path)
            preprocessor = CLIPImageProcessor.from_pretrained(clip_model_path)
            
            # Transfer weights from trained model to HuggingFace model
            print("Transferring weights...")
            self._transfer_weights(vit)
            
            # Save the converted model
            print(f"Saving converted model to: {output_path}")
            os.makedirs(output_path, exist_ok=True)
            
            preprocessor.save_pretrained(output_path)
            vit.save_pretrained(output_path)
            
            # Save additional components
            self._save_additional_components(output_path)
            
            # Save conversion log
            self._save_conversion_log(clip_model_path, output_path)
            
            print("Model conversion completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error during model conversion: {e}")
            return False
            
    def _validate_conversion_paths(self, clip_model_path, output_path):
        """Validate paths for conversion."""
        if not clip_model_path:
            print("Error: CLIP model path not specified in configuration")
            return False
            
        if not output_path:
            print("Error: Output path not specified in configuration")
            return False
            
        if not os.path.exists(clip_model_path):
            print(f"Error: CLIP model not found at: {clip_model_path}")
            return False
            
        return True
        
    def _transfer_weights(self, vit):
        """Transfer weights from trained model to HuggingFace model."""
        try:
            # Get the number of layers (assuming ViT-L with 24 layers)
            num_layers = len(vit.vision_model.encoder.layers)
            
            for i in range(num_layers):
                # Get weights from trained model
                attn_state = self.model.model.model.visual.transformer.resblocks[i].attn.state_dict()
                
                # Extract Q, K, V weights
                in_proj_weight = attn_state["in_proj_weight"]
                embed_dim = in_proj_weight.size(0) // 3
                
                q = in_proj_weight[:embed_dim, :].cpu()
                k = in_proj_weight[embed_dim:embed_dim * 2, :].cpu()
                v = in_proj_weight[embed_dim * 2:embed_dim * 3, :].cpu()
                o = attn_state['out_proj.weight'].cpu()
                
                # Set weights in HuggingFace model
                vit.vision_model.encoder.layers[i].self_attn.q_proj.weight = torch.nn.Parameter(q)
                vit.vision_model.encoder.layers[i].self_attn.k_proj.weight = torch.nn.Parameter(k)
                vit.vision_model.encoder.layers[i].self_attn.v_proj.weight = torch.nn.Parameter(v)
                vit.vision_model.encoder.layers[i].self_attn.out_proj.weight = torch.nn.Parameter(o)
                
        except Exception as e:
            print(f"Error transferring weights: {e}")
            raise
            
    def _save_additional_components(self, output_path):
        """Save additional model components."""
        try:
            # Save projection layer
            proj_path = os.path.join(output_path, "proj.pth")
            torch.save(self.model.model.model.visual.proj, proj_path)
            
            # Save FC layer if available
            checkpoint_path = self.config.get('visual_pretraining.checkpoint_path')
            if checkpoint_path and os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                if isinstance(checkpoint, dict) and 'fc' in checkpoint:
                    fc_path = os.path.join(output_path, "fc.pth")
                    torch.save(checkpoint['fc'], fc_path)
                    
        except Exception as e:
            print(f"Warning: Could not save additional components: {e}")
            
    def _save_conversion_log(self, clip_model_path, output_path):
        """Save conversion log."""
        log_path = os.path.join(output_path, "conversion_log.txt")
        
        with open(log_path, 'w') as f:
            f.write("Model Conversion to HuggingFace Format Log\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Base CLIP model: {clip_model_path}\n")
            f.write(f"Trained checkpoint: {self.config.get('visual_pretraining.checkpoint_path')}\n")
            f.write(f"Output path: {output_path}\n")
            f.write(f"Training mode: {self.config.get('training.trainmode')}\n")
            f.write(f"Model name: {self.config.get('training.modelname')}\n")
            f.write(f"Conversion time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
            f.write("\nConversion completed successfully!\n")
            
    def print_info(self):
        """Print conversion information."""
        print("=" * 80)
        print("Model Conversion Configuration")
        print("=" * 80)
        print(f"CLIP model path: {self.config.get('models.pretrained.clip_model')}")
        print(f"Checkpoint path: {self.config.get('visual_pretraining.checkpoint_path')}")
        print(f"Output path: {self.config.get('models.output.visual_encoder_output')}")
        print(f"Training mode: {self.config.get('training.trainmode')}")
        print(f"Model name: {self.config.get('training.modelname')}")
        print("=" * 80)


def create_convert_argument_parser():
    """Create argument parser for the conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert trained model to HuggingFace format with configuration file support",
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
        '--checkpoint', 
        type=str, 
        help='Path to trained checkpoint (overrides config)'
    )
    parser.add_argument(
        '--clip_model', 
        type=str, 
        help='Path to base CLIP model (overrides config)'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        help='Output path for converted model (overrides config)'
    )
    parser.add_argument(
        '--test_only', 
        action='store_true',
        help='Only test the model without conversion'
    )
    
    return parser


def main():
    """Main function."""
    parser = create_convert_argument_parser()
    args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader(args.config)
    
    # Override with command line arguments if provided
    if args.checkpoint:
        config_loader.set('visual_pretraining.checkpoint_path', args.checkpoint)
    if args.clip_model:
        config_loader.set('models.pretrained.clip_model', args.clip_model)
    if args.output_path:
        config_loader.set('models.output.visual_encoder_output', args.output_path)
    
    # Create converter
    converter = WeightConverter(config_loader)
    converter.print_info()
    
    # Test the model
    converter.model.eval()
    acc, ap = converter.test_model()
    
    print(f"\nModel performance - Accuracy: {acc*100:.1f}%, AP: {ap*100:.1f}%")
    
    if args.test_only:
        print("Test completed (conversion skipped)")
        return
    
    # Convert to HuggingFace format
    success = converter.convert_to_huggingface()
    
    if success:
        print("\nConversion completed successfully!")
        print(f"Converted model saved to: {config_loader.get('models.output.visual_encoder_output')}")
        exit(0)
    else:
        print("\nConversion failed!")
        exit(1)


if __name__ == "__main__":
    main() 