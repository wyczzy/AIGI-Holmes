import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

import os

import torch
from PIL import Image
from tqdm import tqdm

from vllm import LLM, SamplingParams

import torch
from transformers import LogitsProcessor
import math

import torch
from transformers import LogitsProcessor
import math

class ForceLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, target_sequence, real_id, fake_id, logit_real, logit_fake):
        """
        Custom logits processor to force target sequence at the beginning and apply weighting after.

        Args:
            tokenizer (str): Tokenizer name or path.
            target_sequence (list[str]): Target sequence to force at the beginning, e.g., ["This", "is", "a"].
            real_id (int): Token ID for the real token.
            fake_id (int): Token ID for the fake token.
            logit_real (float): Logit bias for the real token.
            logit_fake (float): Logit bias for the fake token.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        # Encode target sequence tokens (without special tokens)
        self.target_tokens = self.tokenizer.encode(" ".join(target_sequence), add_special_tokens=False)
        self.real_id = real_id
        self.fake_id = fake_id
        self.logit_real = logit_real
        self.logit_fake = logit_fake
        self.triggered = False
        self.forced_count = 0  # Track how many tokens we've forced

    def __call__(self, input_ids, scores):
        # Convert input_ids to list if needed
        if isinstance(input_ids, tuple):
            input_ids = list(input_ids)
        elif isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()

        current_length = len(input_ids)
        
        # Phase 1: Force target sequence at the beginning
        if current_length < len(self.target_tokens):
            # Force the next token in the target sequence
            next_token = self.target_tokens[current_length]
            # Set all scores to -inf except the target token
            scores[:] = -float('inf')
            scores[next_token] = 0  # Set to high probability
            self.forced_count = current_length + 1
            return scores
        
        # Phase 2: Apply weighting after target sequence appears
        if not self.triggered:
            # Check if we just finished generating the target sequence
            if current_length >= len(self.target_tokens):
                start_pos = current_length - len(self.target_tokens)
                last_tokens = input_ids[start_pos:start_pos + len(self.target_tokens)]
                
                if last_tokens == self.target_tokens:
                    # Apply logit adjustment for real/fake tokens
                    scores[self.real_id] += self.logit_real
                    scores[self.fake_id] += self.logit_fake
                    self.triggered = True
        
        return scores