from transformers import AutoTokenizer
from logits_processor_zoo.trtllm import ForceLastPhraseLogitsProcessor
from utils import TRTLLMTester, get_parser


if __name__ == "__main__":
    args = get_parser()
    beam_width = 1

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    phrase = "\n\nThanks for trying our application! If you have more questions about"

    lp = ForceLastPhraseLogitsProcessor(phrase, tokenizer, batch_size=1)

    TRTLLMTester(lp, tokenizer, args).run(args.prompt, beam_width)
