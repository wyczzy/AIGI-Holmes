from transformers import AutoTokenizer
from logits_processor_zoo.trtllm import MultipleChoiceLogitsProcessor
from utils import TRTLLMTester, get_parser


if __name__ == "__main__":
    args = get_parser()
    beam_width = 1

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    lp = MultipleChoiceLogitsProcessor(tokenizer, choices=["1", "2"], delimiter=".", boost_first_words=0.5)

    TRTLLMTester(lp, tokenizer, args).run(args.prompt, beam_width, max_new_tokens=1)
