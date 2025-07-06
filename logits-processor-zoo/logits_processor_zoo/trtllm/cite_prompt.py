#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import List, Optional
import torch
from transformers import PreTrainedTokenizer


class CiteFromPromptLogitsProcessor:
    """
    A logits processor which boosts or diminishes the likelihood of tokens present in the prompt (and optionally
    EOS token) to encourage the model to generate tokens similar to those seen in the prompt or vice versa.
    WARNING: Create a new object before every model.generate call since every batch has different prompts.

    Parameters
    ----------
    tokenizer (PreTrainedTokenizer): The tokenizer used by the LLM.
    prompts (List[str]): Prompts in the batch.
    boost_factor (float): A factor to boost the likelihood of the tokens from the prompt.
                            Negative values are used for the opposite effect.
    boost_eos (bool, optional): If True, boosts EOS token too.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, prompts: List[str], boost_factor: float = 1.0,
                 boost_eos: bool = True):
        self.boost_factor = boost_factor

        self.boost_ids = []
        for prompt in prompts:
            prompt_tokens = set(tokenizer.encode(prompt))

            if boost_eos:
                prompt_tokens.add(tokenizer.eos_token_id)

            self.boost_ids.append(list(prompt_tokens))

    def __call__(self, req_ids_batch: List[int], logits_batch: List[torch.Tensor],
                 ids_batch: List[List[List[int]]], stream_ptr,
                 client_ids_batch: List[Optional[int]]):

        with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
            for i in range(logits_batch.shape[1]):
                logits_batch[:, i, self.boost_ids[i]] += self.boost_factor
