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
from transformers import PreTrainedTokenizer
import torch
from logits_processor_zoo.utils import text_to_token


class GenLengthLogitsProcessor:
    """
    A logits processor that adjusts the likelihood of the end-of-sequence (EOS) token
    based on the length of the generated sequence, encouraging or discouraging shorter answers.
    WARNING: Create a new object before every model.generate call since token_count is accumulated.

    Parameters
    ----------
    tokenizer (PreTrainedTokenizer): The tokenizer used by the LLM.
    boost_factor (float): A factor to boost the likelihood of the EOS token as the sequence length increases.
                        Suggested value range is [-1.0, 1.0]. Negative values are used for the opposite effect.
    p (int, optional): The power to which the token count is raised when computing the boost value. Default is 2.
    complete_sentences (bool, optional): If True, boosts EOS token likelihood only when the last token is a full stop
                                        or a new line. Default is False.

    """

    def __init__(self, tokenizer: PreTrainedTokenizer, boost_factor: float,
                 p: int = 2, complete_sentences: bool = False):
        self.eos_token = tokenizer.eos_token_id
        self.boost_factor = boost_factor
        self.p = p
        self.token_count = 0
        self.full_stop_token = text_to_token(tokenizer, "It is a sentence.", last=True)
        self.new_line_token = text_to_token(tokenizer, "It is a new line\n", last=True)
        self.complete_sentences = complete_sentences

    def __call__(self, req_ids_batch: List[int], logits_batch: List[torch.Tensor],
                 ids_batch: List[List[List[int]]], stream_ptr,
                 client_ids_batch: List[Optional[int]]):

        boost_val = self.boost_factor * (self.token_count ** self.p) / (10 ** self.p)

        with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
            ids_batch = torch.LongTensor(ids_batch).to(logits_batch.device, non_blocking=True)

            if self.complete_sentences:
                enabled = (ids_batch[:, -1] == self.full_stop_token) | (ids_batch[:, -1] == self.new_line_token)
                logits_batch[:, :, self.eos_token] += enabled * boost_val
            else:
                logits_batch[:, :, self.eos_token] += boost_val

        self.token_count += 1
