"""
Enhanced transform_vision_model.py with configuration file support.
Replaces the vision model in LLaVA with a trained visual encoder.
"""

import os
import argparse
from config_loader import ConfigLoader, create_argument_parser, load_config_from_args
from transformers import CLIPVisionModel, CLIPImageProcessor, LlavaNextForConditionalGeneration, LlamaTokenizerFast, LlavaNextImageProcessor, LlavaNextProcessor


class VisionModelTransformer:
    """
    Handles the transformation of vision models in LLaVA.
    """
    
    def __init__(self, config_loader: ConfigLoader):
        """
        Initialize the vision model transformer.
        
        Args:
            config_loader: ConfigLoader instance containing all configuration
        """
        self.config = config_loader
        
    def transform_vision_model(self):
        """
        Transform the vision model in LLaVA by replacing it with a trained visual encoder.
        """
        print("Starting vision model transformation...")
        
        # Get paths from configuration
        llava_model_path = self.config.get('models.pretrained.llava_model')
        visual_encoder_path = self.config.get('models.output.visual_encoder_output')
        output_path = self.config.get('models.output.llava_output')
        
        # Validate paths
        if not self._validate_paths(llava_model_path, visual_encoder_path, output_path):
            return False
            
        try:
            # Load LLaVA model
            print(f"Loading LLaVA model from: {llava_model_path}")
            llava_model = LlavaNextForConditionalGeneration.from_pretrained(llava_model_path)
            tokenizer = LlamaTokenizerFast.from_pretrained(llava_model_path, trust_remote_code=False)
            
            # Load trained visual encoder
            print(f"Loading visual encoder from: {visual_encoder_path}")
            visual_encoder = CLIPVisionModel.from_pretrained(visual_encoder_path)
            image_processor = LlavaNextImageProcessor.from_pretrained(visual_encoder_path)
            
            # Replace vision tower
            print("Replacing vision tower...")
            llava_model.vision_tower = visual_encoder
            
            # Create processor
            processor = LlavaNextProcessor(tokenizer=tokenizer, image_processor=image_processor)
            
            # Save the modified model
            print(f"Saving modified LLaVA model to: {output_path}")
            os.makedirs(output_path, exist_ok=True)
            llava_model.save_pretrained(output_path)
            processor.save_pretrained(output_path)
            
            print("Vision model transformation completed successfully!")
            
            # Save transformation log
            self._save_transformation_log(llava_model_path, visual_encoder_path, output_path)
            
            return True
            
        except Exception as e:
            print(f"Error during vision model transformation: {e}")
            return False
            
    def _validate_paths(self, llava_model_path, visual_encoder_path, output_path):
        """
        Validate the required paths.
        
        Args:
            llava_model_path: Path to the LLaVA model
            visual_encoder_path: Path to the visual encoder
            output_path: Path to save the output
            
        Returns:
            bool: True if all paths are valid
        """
        if not llava_model_path:
            print("Error: LLaVA model path not specified in configuration")
            return False
            
        if not visual_encoder_path:
            print("Error: Visual encoder path not specified in configuration")
            return False
            
        if not output_path:
            print("Error: Output path not specified in configuration")
            return False
            
        if not os.path.exists(llava_model_path):
            print(f"Error: LLaVA model not found at: {llava_model_path}")
            return False
            
        if not os.path.exists(visual_encoder_path):
            print(f"Error: Visual encoder not found at: {visual_encoder_path}")
            return False
            
        return True
        
    def _save_transformation_log(self, llava_model_path, visual_encoder_path, output_path):
        """
        Save a log of the transformation process.
        
        Args:
            llava_model_path: Path to the original LLaVA model
            visual_encoder_path: Path to the visual encoder
            output_path: Path to the output model
        """
        log_path = os.path.join(output_path, "transformation_log.txt")
        
        with open(log_path, 'w') as f:
            f.write("LLaVA Vision Model Transformation Log\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Original LLaVA model: {llava_model_path}\n")
            f.write(f"Visual encoder: {visual_encoder_path}\n")
            f.write(f"Output model: {output_path}\n")
            f.write(f"Transformation time: {os.popen('date').read().strip()}\n")
            f.write("\nTransformation completed successfully!\n")
            
    def print_info(self):
        """Print transformation information."""
        print("=" * 80)
        print("LLaVA Vision Model Transformation Configuration")
        print("=" * 80)
        print(f"LLaVA model path: {self.config.get('models.pretrained.llava_model')}")
        print(f"Visual encoder path: {self.config.get('models.output.visual_encoder_output')}")
        print(f"Output path: {self.config.get('models.output.llava_output')}")
        print("=" * 80)


def create_transform_argument_parser():
    """
    Create argument parser for the transform script.
    
    Returns:
        ArgumentParser object
    """
    parser = argparse.ArgumentParser(
        description="Transform LLaVA vision model with configuration file support",
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
        '--llava_model', 
        type=str, 
        help='Path to LLaVA model (overrides config)'
    )
    parser.add_argument(
        '--visual_encoder', 
        type=str, 
        help='Path to visual encoder (overrides config)'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        help='Output path for transformed model (overrides config)'
    )
    
    return parser


def main():
    """Main function."""
    parser = create_transform_argument_parser()
    args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader(args.config)
    
    # Override with command line arguments if provided
    if args.llava_model:
        config_loader.set('models.pretrained.llava_model', args.llava_model)
    if args.visual_encoder:
        config_loader.set('models.output.visual_encoder_output', args.visual_encoder)
    if args.output_path:
        config_loader.set('models.output.llava_output', args.output_path)
    
    # Create and run transformer
    transformer = VisionModelTransformer(config_loader)
    transformer.print_info()
    
    success = transformer.transform_vision_model()
    
    if success:
        print("\nTransformation completed successfully!")
        exit(0)
    else:
        print("\nTransformation failed!")
        exit(1)


if __name__ == "__main__":
    main() 