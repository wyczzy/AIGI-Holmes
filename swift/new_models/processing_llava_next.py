from typing import List, Union
from PIL import Image
import numpy as np
import torch
from transformers import BatchFeature
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.image_utils import ImageInput, get_image_size, to_numpy_array
from transformers.models.llava_next.processing_llava_next import LlavaNextProcessorKwargs
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, _validate_images_text_input_order
from transformers import AutoProcessor


from transformers.models.llava_next import LlavaNextProcessor

class CustomLlavaNextProcessor(LlavaNextProcessor):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size=None,
        vision_feature_select_strategy=None,
        chat_template=None,
        image_token="<image>",
        **kwargs,
    ):
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            patch_size=patch_size,
            vision_feature_select_strategy=vision_feature_select_strategy,
            chat_template=chat_template,
            image_token=image_token,
            **kwargs,
        )
        self.vision_feature_select_strategy = 'expert'

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[LlavaNextProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        LlavaNextImageProcessor's [`~LlavaNextImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if images is None and text is None:
            raise ValueError("You have to specify at least images or text.")
        # check if images and text inputs are reversed for BC
        images, text = _validate_images_text_input_order(images, text)

        output_kwargs = self._merge_kwargs(
            LlavaNextProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        prompt_strings = text
        if image_inputs:
            if self.patch_size is None or self.vision_feature_select_strategy is None:
                pass
                # logger.warning_once(
                #     "Expanding inputs for image tokens in LLaVa-NeXT should be done in processing. "
                #     "Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly "
                #     "with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. "
                #     "Using processors without these attributes in the config is deprecated and will throw an error in v4.47."
                # )
            else:
                image_sizes = iter(image_inputs["image_sizes"])
                b, p, c, h, w = image_inputs["pixel_values"].shape
                height, width = get_image_size(to_numpy_array(image_inputs["pixel_values"][0][0]))
                prompt_strings = []
                for sample in text:
                    while self.image_token in sample:
                        image_size = next(image_sizes)
                        orig_height, orig_width = image_size
                        num_image_tokens = self._get_number_of_features(orig_height, orig_width, height, width)
                        if self.vision_feature_select_strategy == "default":
                            num_image_tokens -= 1
                        elif self.vision_feature_select_strategy == "expert":
                            num_image_tokens += (p*256-1)
                        sample = sample.replace(self.image_token, "<placeholder>" * num_image_tokens, 1)
                    prompt_strings.append(sample)
                prompt_strings = [sample.replace("<placeholder>", self.image_token) for sample in prompt_strings]

        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs})

AutoProcessor.register("llava_with_vision_expert", CustomLlavaNextProcessor, exist_ok=True)
if __name__ == '__main__':
    processor = CustomLlavaNextProcessor.from_pretrained("/data/kesun/kesun/llava-v1.6-mistral-7b-hf-with-expert-1-ours")
    processor = AutoProcessor.from_pretrained("/data/kesun/kesun/llava-v1.6-mistral-7b-hf-with-expert-1-ours")
    print(processor)