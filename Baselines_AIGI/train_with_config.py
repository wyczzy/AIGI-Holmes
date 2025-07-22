"""
Unified training script with configuration support.
Supports visual pretraining and automatic model conversion.
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
from config_loader import ConfigLoader, create_argument_parser, load_config_from_args
from transformers import CLIPVisionModel, CLIPImageProcessor
from transformers import CLIPVisionModel, CLIPImageProcessor, LlavaNextForConditionalGeneration, LlamaTokenizerFast, LlavaNextImageProcessor, LlavaNextProcessor

import random
import subprocess
import shutil


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


class UnifiedTrainer:
    """
    Unified trainer that handles the complete training pipeline including:
    - Visual pretraining
    - Model conversion to HuggingFace format
    - Vision model replacement in LLaVA
    """
    
    def __init__(self, config_loader: ConfigLoader):
        """
        Initialize the unified trainer.
        
        Args:
            config_loader: ConfigLoader instance containing all configuration
        """
        self.config = config_loader
        self.setup_environment()
        self.setup_data_and_model()
        
    def setup_environment(self):
        """Setup training environment."""
        seed_torch(self.config.get('training.seed', 100))
        
        # Create necessary directories
        self.config.create_directories()
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging and tensorboard."""
        log_dir = os.path.join(
            self.config.get('models.checkpoints_dir'), 
            self.config.get('training.name')
        )
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup file logger
        Logger(os.path.join(log_dir, self.config.get('logging.log_file', 'training.log')))
        
        # Setup tensorboard writers
        if self.config.get('logging.tensorboard', True):
            self.train_writer = SummaryWriter(os.path.join(log_dir, "train"))
            self.val_writer = SummaryWriter(os.path.join(log_dir, "val"))
        else:
            self.train_writer = None
            self.val_writer = None
            
    def setup_data_and_model(self):
        """Setup data loaders and model."""
        # Convert config to namespace for compatibility
        self.opt = self.config.to_namespace()
        
        # Update specific paths
        self.opt.dataroot = f"{self.config.get('data.train_dataroot')}/{self.config.get('data.train_split')}"
        self.test_dataroot = self.config.get('data.test_dataroot')
        
        # Create data loader
        self.data_loader = create_dataloader(self.opt)
        
        # Create model
        self.model = Trainer(self.opt)
        
        # Load pretrained model if specified
        checkpoint_path = self.config.get('visual_pretraining.checkpoint_path')
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading pretrained model from: {checkpoint_path}")
            self.model.model.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu"), 
                strict=False
            )
            
    def create_test_options(self):
        """Create test options from configuration."""
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
        return test_opt
        
    def test_model(self):
        """Test the model on all configured test datasets."""
        print('*' * 25)
        accs = []
        aps = []
        
        test_opt = self.create_test_options()
        test_vals = self.config.get('testing.test_vals', [])
        multiclass = self.config.get('testing.multiclass', [])
        
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        
        for v_id, val in enumerate(test_vals):
            test_opt.dataroot = f'{self.test_dataroot}/{val}'
            test_opt.classes = os.listdir(test_opt.dataroot) if (v_id < len(multiclass) and multiclass[v_id]) else ['']
            
            acc, ap, r_acc, f_acc, _, _ = validate(self.model.model, test_opt)
            accs.append(acc)
            aps.append(ap)
            
            print(f"({v_id} {val:>10}) acc: {acc*100:.1f}; ap: {ap*100:.1f}, racc: {r_acc*100:.1f}, facc: {f_acc*100:.1f};")
            
        print(f"({len(test_vals)} {'Mean':>10}) acc: {np.array(accs).mean()*100:.1f}; ap: {np.array(aps).mean()*100:.1f}")
        print('*' * 25)
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        
        return np.mean(accs), np.mean(aps)
        
    def train(self):
        """Execute the training loop."""
        print(f"Starting training with config: {self.config.config_path}")
        if self.config.get('logging.print_options', True):
            self.config.print_config()
            
        # Initial testing
        print("Initial model testing...")
        # self.model.eval()
        # self.test_model()
        self.model.train()
        
        print(f"Current working directory: {os.getcwd()}")
        
        # Training loop
        for epoch in range(self.config.get('training.niter', 1000)):
            epoch_start_time = time.time()
            epoch_iter = 0
            
            print(f"Epoch {epoch + 1}/{self.config.get('training.niter', 1000)}")
            
            for i, data in enumerate(self.data_loader):
                self.model.total_steps += 1
                epoch_iter += self.config.get('training.batch_size', 16)
                
                self.model.set_input(data)
                self.model.optimize_parameters()
                
                # Log training loss
                if self.model.total_steps % self.config.get('training.loss_freq', 400) == 0:
                    print(f"{time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())} "
                          f"Train loss: {self.model.loss} at step: {self.model.total_steps} lr {self.model.lr}")
                    
                    if self.train_writer:
                        self.train_writer.add_scalar('loss', self.model.loss, self.model.total_steps)
                        
            # Learning rate decay
            if epoch % self.config.get('training.delr_freq', 20) == 0 and epoch != 0:
                print(f"{time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())} "
                      f"Changing lr at the end of epoch {epoch}, iters {self.model.total_steps}")
                self.model.adjust_learning_rate()
                
            # Validation
            self.model.eval()
            acc, ap = self.test_model()
            self.model.save_networks(f'{acc:.2f}_{ap:.2f}')
            
            if self.val_writer:
                self.val_writer.add_scalar('accuracy', acc, self.model.total_steps)
                self.val_writer.add_scalar('ap', ap, self.model.total_steps)
                
            print(f"(Val @ epoch {epoch}) acc: {acc}; ap: {ap}")
            self.model.train()
            
        # Final testing and saving
        self.model.eval()
        self.test_model()
        self.model.save_networks('last')
        
        print("Training completed!")
        
    def convert_to_huggingface(self):
        """Convert the trained model to HuggingFace format."""
        if not self.config.get('visual_pretraining.auto_convert_to_hf', False):
            print("Auto conversion to HuggingFace format is disabled.")
            return
            
        print("Converting model to HuggingFace format...")
        
        # Load the base CLIP model
        clip_model_path = self.config.get('models.pretrained.clip_model')
        if not clip_model_path or not os.path.exists(clip_model_path):
            print(f"Warning: CLIP model not found at {clip_model_path}")
            return
            
        try:
            vit = CLIPVisionModel.from_pretrained(clip_model_path)
            preprocessor = CLIPImageProcessor.from_pretrained(clip_model_path)
            
            # Transfer weights from trained model to HuggingFace model
            for i in range(24):  # Assuming 24 layers for ViT-L
                # Get weights from trained model
                attn_state = self.model.model.model.visual.transformer.resblocks[i].attn.state_dict()
                
                q = attn_state["in_proj_weight"][:1024, :].cpu()
                k = attn_state["in_proj_weight"][1024:1024 * 2, :].cpu()
                v = attn_state["in_proj_weight"][1024 * 2:1024 * 3, :].cpu()
                o = attn_state['out_proj.weight'].cpu()
                
                # Set weights in HuggingFace model
                vit.vision_model.encoder.layers[i].self_attn.q_proj.weight = torch.nn.Parameter(q)
                vit.vision_model.encoder.layers[i].self_attn.k_proj.weight = torch.nn.Parameter(k)
                vit.vision_model.encoder.layers[i].self_attn.v_proj.weight = torch.nn.Parameter(v)
                vit.vision_model.encoder.layers[i].self_attn.out_proj.weight = torch.nn.Parameter(o)
                
            # Save the converted model
            output_path = self.config.get('models.output.visual_encoder_output')
            os.makedirs(output_path, exist_ok=True)
            
            preprocessor.save_pretrained(output_path)
            vit.save_pretrained(output_path)
            
            # Save additional components
            proj_path = os.path.join(output_path, "proj.pth")
            fc_path = os.path.join(output_path, "fc.pth")
            
            torch.save(self.model.model.model.visual.proj, proj_path)
            
            # Save the final checkpoint
            checkpoint_path = self.config.get('visual_pretraining.checkpoint_path')
            if checkpoint_path and os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                if 'fc' in checkpoint:
                    torch.save(checkpoint['fc'], fc_path)
                    
            print(f"Model converted to HuggingFace format and saved to: {output_path}")
            
        except Exception as e:
            print(f"Error converting model to HuggingFace format: {e}")
            
    def replace_llava_vision_model(self):
        """Replace the vision model in LLaVA with the trained model."""
        if not self.config.get('visual_pretraining.auto_replace_llava_vision', False):
            print("Auto replacement of LLaVA vision model is disabled.")
            return
            
        print("Replacing LLaVA vision model with trained model...")
        
        llava_model_path = self.config.get('models.pretrained.llava_model')
        visual_encoder_path = self.config.get('models.output.visual_encoder_output')
        output_path = self.config.get('models.output.llava_output')
        
        if not all([llava_model_path, visual_encoder_path, output_path]):
            print("Missing required paths for LLaVA vision model replacement.")
            return
            
        if not os.path.exists(llava_model_path):
            print(f"LLaVA model not found at: {llava_model_path}")
            return
            
        if not os.path.exists(visual_encoder_path):
            print(f"Visual encoder not found at: {visual_encoder_path}")
            return
            
        try:
            # Load LLaVA model
            llava_model = LlavaNextForConditionalGeneration.from_pretrained(llava_model_path)
            tokenizer = LlamaTokenizerFast.from_pretrained(llava_model_path, trust_remote_code=False)
            
            # Load trained visual encoder
            visual_encoder = CLIPVisionModel.from_pretrained(visual_encoder_path)
            image_processor = LlavaNextImageProcessor.from_pretrained(visual_encoder_path)
            
            # Replace vision tower
            llava_model.vision_tower = visual_encoder
            
            # Create processor
            processor = LlavaNextProcessor(tokenizer=tokenizer, image_processor=image_processor)
            
            # Save the modified model
            os.makedirs(output_path, exist_ok=True)
            llava_model.save_pretrained(output_path)
            processor.save_pretrained(output_path)
            
            print(f"LLaVA model with replaced vision encoder saved to: {output_path}")
            
        except Exception as e:
            print(f"Error replacing LLaVA vision model: {e}")
            
    def run_complete_pipeline(self):
        """Run the complete training and conversion pipeline."""
        print("=" * 80)
        print("Starting complete AIGC detection training pipeline")
        print("=" * 80)
        
        # Step 1: Training
        print("\n" + "=" * 40)
        print("Step 1: Visual Pretraining")
        print("=" * 40)
        self.train()
        
        # Step 2: Convert to HuggingFace format
        print("\n" + "=" * 40)
        print("Step 2: Converting to HuggingFace format")
        print("=" * 40)
        self.convert_to_huggingface()
        
        # Step 3: Replace LLaVA vision model
        print("\n" + "=" * 40)
        print("Step 3: Replacing LLaVA vision model")
        print("=" * 40)
        self.replace_llava_vision_model()
        
        print("\n" + "=" * 80)
        print("Complete pipeline finished!")
        print("=" * 80)
        
        # Print summary
        print(f"Checkpoints saved to: {self.config.get('models.checkpoints_dir')}")
        print(f"HuggingFace model saved to: {self.config.get('models.output.visual_encoder_output')}")
        print(f"Modified LLaVA model saved to: {self.config.get('models.output.llava_output')}")


def main():
    """Main function."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Load configuration
    config_loader = load_config_from_args(args)
    
    # Create and run trainer
    trainer = UnifiedTrainer(config_loader)
    trainer.run_complete_pipeline()


if __name__ == '__main__':
    main() 