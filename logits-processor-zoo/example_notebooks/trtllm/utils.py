import argparse
import datetime
from typing import List

import tensorrt_llm.bindings.executor as trtllm


# TensorRT-LLM utility functions are taken from:
# https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/bindings/executor/example_logits_processor.py
# Prepare and enqueue the requests
class TRTLLMTester:
    def __init__(self, logits_processor, tokenizer, args):
        self.logits_processor = logits_processor
        self.tokenizer = tokenizer
        self.args = args

    def enqueue_requests(self, prompt: List[int], executor: trtllm.Executor,
                         beam_width: int, max_new_tokens: int, batch_size: int = 1):
        sampling_config = trtllm.SamplingConfig(beam_width)

        request_ids = []
        for iter_id in range(batch_size):
            # Create the request.
            request = trtllm.Request(input_token_ids=prompt,
                                     max_new_tokens=max_new_tokens,
                                     end_id=self.tokenizer.eos_token_id,
                                     sampling_config=sampling_config,
                                     client_id=iter_id % 2)
            request.logits_post_processor_name = "my_logits_pp"

            # Enqueue the request.
            req_id = executor.enqueue_request(request)
            request_ids.append(req_id)

        return request_ids

    # Wait for responses and store output tokens
    def wait_for_responses(self, request_ids: List[int],
                           executor: trtllm.Executor, beam_width: int):
        output_tokens = {
            req_id: {beam: []
                     for beam in range(beam_width)}
            for req_id in request_ids
        }
        num_finished = 0
        iter = 0
        while num_finished < len(request_ids) and iter < self.args.timeout_ms:
            responses = executor.await_responses(
                datetime.timedelta(milliseconds=self.args.timeout_ms))
            for response in responses:
                req_id = response.request_id
                if not response.has_error():
                    result = response.result
                    num_finished += 1 if result.is_final else 0
                    for beam, outTokens in enumerate(result.output_token_ids):
                        output_tokens[req_id][beam].extend(outTokens)
                else:
                    raise RuntimeError(f"{req_id} encountered error: {response.error_msg}")

        return output_tokens

    def run(self, prompt: str, beam_width: int = 1, max_new_tokens: int = 2000):
        # Create the executor.
        executor_config = trtllm.ExecutorConfig(beam_width)
        executor_config.logits_post_processor_map = {
            "my_logits_pp": self.logits_processor
        }
        executor = trtllm.Executor(self.args.engine_path, trtllm.ModelType.DECODER_ONLY,
                                   executor_config)

        prompt_encoded = self.tokenizer.encode(prompt)
        print(f"Input text: {prompt}\n")

        if executor.can_enqueue_requests():
            request_ids = self.enqueue_requests(prompt_encoded, executor, beam_width, max_new_tokens)
            output_tokens = self.wait_for_responses(request_ids, executor, beam_width)

            # Print output
            for req_id in request_ids:
                for beam_id in range(beam_width):
                    result = self.tokenizer.decode(
                        output_tokens[req_id][beam_id][len(prompt_encoded):])
                    generated_tokens = len(
                        output_tokens[req_id][beam_id]) - len(prompt_encoded)
                    print(
                        f"Request {req_id} Beam {beam_id} ({generated_tokens} tokens): {result}"
                    )


def get_parser():
    parser = argparse.ArgumentParser(description="Logits Processor Example")
    parser.add_argument("--tokenizer_path",
                        "-t",
                        type=str,
                        required=True,
                        help="Directory containing model tokenizer")
    parser.add_argument("--engine_path",
                        "-e",
                        type=str,
                        required=True,
                        help="Directory containing model engine")
    parser.add_argument("--prompt",
                        "-p",
                        type=str,
                        default="Please give me information about macaques:",
                        help="Prompt to test")
    parser.add_argument(
        "--timeout_ms",
        type=int,
        required=False,
        default=10000,
        help="The maximum time to wait for all responses, in milliseconds")

    return parser.parse_args()
