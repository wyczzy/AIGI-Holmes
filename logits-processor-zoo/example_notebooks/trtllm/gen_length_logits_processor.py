from transformers import AutoTokenizer
from logits_processor_zoo.trtllm import GenLengthLogitsProcessor
from utils import TRTLLMTester, get_parser


if __name__ == "__main__":
    args = get_parser()
    beam_width = 1

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    lp = GenLengthLogitsProcessor(tokenizer, boost_factor=1.0, complete_sentences=True)

    TRTLLMTester(lp, tokenizer, args).run(args.prompt, beam_width)
