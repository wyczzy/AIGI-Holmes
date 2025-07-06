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

class CustomLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, target_sequence, real_id, fake_id, logit_real, logit_fake):
        """
        Custom logits processor to apply weighting after the target sequence.

        Args:
            tokenizer (PreTrainedTokenizer): The tokenizer used for encoding texts.
            target_sequence (list[str]): The target sequence to check, e.g., ["This", "is", "a"].
            real_id (int): The token ID for the real token.
            fake_id (int): The token ID for the fake token.
            logit_real (float): Logit bias for the real token.
            logit_fake (float): Logit bias for the fake token.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.target_tokens = self.tokenizer.encode(" ".join(target_sequence), add_special_tokens=False)
        self.real_id = real_id
        self.fake_id = fake_id
        self.logit_real = logit_real
        self.logit_fake = logit_fake
        self.triggered = False  # 状态变量，记录是否已经处理过

    def __call__(self, input_ids, scores):
        """
        Apply custom logits processing.

        Args:
            input_ids (torch.Tensor or list): Generated token IDs so far.
            scores (torch.Tensor): scores distribution for the next token.

        Returns:
            torch.Tensor: Modified scores tensor.
        """
        # Convert input_ids to a list if it's a tuple or tensor
        if isinstance(input_ids, tuple):
            input_ids = list(input_ids)
        elif isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()

        # 只在第一次出现时处理
        if (not self.triggered) and len(input_ids) >= len(self.target_tokens):
            last_tokens = input_ids[-len(self.target_tokens):]
            if last_tokens == self.target_tokens:
                scores[self.real_id] += self.logit_real
                scores[self.fake_id] += self.logit_fake
                self.triggered = True  # 标记已处理
        return scores