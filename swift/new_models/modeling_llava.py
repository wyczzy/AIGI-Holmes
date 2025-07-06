from transformers import LlavaForConditionalGeneration, LlavaNextForConditionalGeneration
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast, LLAVA_INPUTS_DOCSTRING
from transformers.models.llava_next.modeling_llava_next import LlavaNextCausalLMOutputWithPast
from transformers.models.llava_next.modeling_llava_next import get_anyres_image_grid_shape, unpad_image, image_size_to_num_patches

from transformers import AutoModel
import torch
from typing import Optional, Union, Tuple, List
import torch.nn as nn
from swift.new_models.config_llava_new import LlavaWithResNetConfig, LlavaWithVisionExpertConfig, ResnetConfig
# from models.config_llava_new import LlavaWithResNetConfig, LlavaWithVisionExpertConfig, ResnetConfig
from swift.new_models.resnet import resnet50
import torch.nn.functional as F

from transformers import PreTrainedModel


class ResnetExpertModel(PreTrainedModel):
    config_class = ResnetConfig
    supports_gradient_checkpointing = True
    _no_split_modules = []

    def __init__(self, config: ResnetConfig):
        super().__init__(config)
        self.model = resnet50(
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            use_low_level=config.use_low_level
        )
        if config.pretrain_path != "":
            self.pretrained_weights = torch.load(config.pretrain_path, map_location='cpu')

            self.model.load_state_dict(self.pretrained_weights, strict=False)

    def forward(self, tensor):
        return self.model.forward_features(tensor)


class CustomLlavaNextForConditionalGeneration(LlavaNextForConditionalGeneration):
    config_class = LlavaWithVisionExpertConfig

    def __init__(self, config: LlavaWithVisionExpertConfig):
        super().__init__(config)
        self.vision_expert = AutoModel.from_config(config.expert_config)

    def pack_image_features(self, image_features, image_sizes, vision_feature_select_strategy, add_image_features=None, image_newline=None):
        """
        Reshape, unpad and then pack each image_feature into a single image_features tensor containing all visual vectors.

        Args:
            image_features (`List[torch.Tensor]` of length num_images, each of shape `(num_patches, image_length, embed_dim)`)
                List of image feature tensor, each contains all the visual feature of all patches.
            image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
                Actual image size of each images (H, W).
            vision_feature_select_strategy (`str`)
                The feature selection strategy used to select the vision feature from the vision backbone.
            image_newline (`torch.Tensor` of shape `(embed_dim)`)
                New line embedding vector.
        Returns:
            image_features (`torch.Tensor` of shape `(all_feat_len, embed_dim)`)
            feature_lens (`List[int]`)
                token length of each image in image_features
        """
        new_image_features = []
        # num_patch = image_features.shape[0]
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size

                if vision_feature_select_strategy == "default":
                    expected_num_patches = height * width
                elif vision_feature_select_strategy == "full":
                    expected_num_patches = height * width + 1
                elif vision_feature_select_strategy == "expert":
                    expected_num_patches = height * width
                if expected_num_patches != base_image_feature.shape[0]:
                    raise ValueError("The number of patches is not consistent with the image size.")

                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                if image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = torch.cat((image_feature, image_newline[None].to(image_feature)), dim=0)
            new_image_features.append(torch.cat((add_image_features[image_idx].reshape((-1, image_feature.shape[-1])), image_feature), dim=0))
            # new_image_features.append(image_feature)
            feature_lens.append(add_image_features[image_idx].reshape((-1, image_feature.shape[-1])).size(0) + image_feature.size(0))
        image_features = torch.cat(new_image_features, dim=0)
        feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features.device)
        return image_features, feature_lens

    def get_image_features(
            self,
            pixel_values: torch.FloatTensor,
            image_sizes: torch.Tensor,
            vision_feature_layer: int,
            vision_feature_select_strategy: str,
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, num_patches, channels, height, width)`)
               The tensors corresponding to the input images.
            image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
                Actual image size of each images (H, W).
            vision_feature_layer (`int`):
                The index of the layer to select the vision feature.
            vision_feature_select_strategy (`str`):
                The feature selection strategy used to select the vision feature from the vision backbone.
                Can be one of `"default"` or `"full"`
        Returns:
            image_features (List[`torch.Tensor`]): List of image feature tensor, each contains all the visual feature of all patches
            and are of shape `(num_patches, image_length, embed_dim)`).
        """
        # ! infer image_num_patches from image_sizes
        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            )
            for imsize in image_sizes
        ]
        if pixel_values.dim() == 5:
            # stacked if input is (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

        image_features = self.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_features.hidden_states[vision_feature_layer]
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        elif vision_feature_select_strategy == "expert":
            selected_image_feature = selected_image_feature[:, 1:]

            add_image_features = self.vision_expert(F.interpolate(pixel_values, size=(224, 224), mode='bilinear'))['last_hidden_state'][:, 1:]

            # add_image_features = add_image_features.unsqueeze(1)
            add_image_features = add_image_features.to(selected_image_feature.device, selected_image_feature.dtype)

            # selected_image_feature = torch.cat((add_image_features, selected_image_feature), dim=1)

        image_features = self.multi_modal_projector(selected_image_feature)
        add_image_features = self.multi_modal_projector(add_image_features)
        image_features = torch.split(image_features, image_num_patches, dim=0)
        add_image_features = torch.split(add_image_features, image_num_patches, dim=0)
        return image_features, add_image_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_sizes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, LlavaNextCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaNextForConditionalGeneration

        >>> model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

        >>> prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "[INST]  \nWhat is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multi-dimensional plot (...)"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        legacy_processing = False
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # if the number of image tokens is more than image embeddings seq length, then prob we expanded it in processing
            # not very reliable, but we don't expect one to actually pass 500+ images for one prompt
            # In case we're in decoding stage, legacy behavior is checked by presence of pixel values even if use_cache=True
            legacy_processing = (
                (input_ids == self.config.image_token_index).sum(1).max() < self.config.image_seq_length
            ) or (input_ids.shape[-1] == 1 and pixel_values is not None)

        image_features = None
        if pixel_values is not None and pixel_values.size(0) > 0:
            image_features, add_image_features = self.get_image_features(
                pixel_values,
                image_sizes,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )

            # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"
            image_features, feature_lens = self.pack_image_features(
                image_features,
                image_sizes,
                add_image_features=add_image_features,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_newline=self.image_newline,
            )



        if legacy_processing:
            # logger.warning_once(
            #     "Expanding inputs for image tokens in LLaVa-NeXT should be done in processing. "
            #     "Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly "
            #     "with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. "
            #     "Using processors without these attributes in the config is deprecated and will throw an error in v4.47."
            # )
            if input_ids.shape[1] != 1:
                image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
                # Add image feature
                # add_image_features = self.multi_modal_projector(self.vision_expert(pixel_values))
                # add_image_features = add_image_features.to(inputs_embeds.device, inputs_embeds.dtype)
                #
                # image_features = torch.cat((add_image_features, image_features), dim=0)
                #
                # inputs_embeds = inputs_embeds.to(image_features.dtype)
                #
                # feature_lens += pixel_values.shape[0]

                inputs_embeds, attention_mask, position_ids, labels, _ = self._merge_input_ids_with_image_features(
                    image_features,
                    feature_lens,
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                    position_ids,
                    labels=labels,
                )
                cache_position = torch.arange(attention_mask.shape[1], device=attention_mask.device)
            else:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0
                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                cache_position = torch.arange(attention_mask.shape[1], device=attention_mask.device)[-target_length:]

        # TODO: @raushan retain only the new behavior after v4.47
        elif image_features is not None:
            n_image_tokens = (input_ids == self.config.image_token_index).sum().item()
            n_image_features = image_features.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            special_image_mask = (
                (input_ids == self.config.image_token_index)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )

            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaNextCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


class CustomLlavaForConditionalGeneration(LlavaForConditionalGeneration):
    config_class = LlavaWithResNetConfig
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    # @add_start_docstrings_to_model_forward(LLAVA_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=LlavaCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            vision_feature_layer: Optional[int] = None,
            vision_feature_select_strategy: Optional[str] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            num_logits_to_keep: int = 0,
    ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.


        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

        >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        legacy_processing = False
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # if the number of image tokens is more than image embeddings seq length, then prob we expanded it in processing
            # not very reliable, but we don't expect one to actually pass 500+ images for one prompt
            # In case we're in decoding stage, legacy behavior is checked by presence of pixel values even if use_cache=True
            legacy_processing = (
                                        (input_ids == self.config.image_token_index).sum(
                                            1).max() < self.config.image_seq_length
                                ) or (input_ids.shape[-1] == 1 and pixel_values is not None)

        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )

        if legacy_processing:
            # prefill stage vs decoding stage (legacy behavior copied)
            if input_ids.shape[1] != 1:
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )
                cache_position = torch.arange(attention_mask.shape[1], device=attention_mask.device)
            else:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                cache_position = torch.arange(attention_mask.shape[1], device=attention_mask.device)[-target_length:]

        # TODO: @raushan retain only the new behavior after v4.47
        elif image_features is not None:
            n_image_tokens = (input_ids == self.config.image_token_index).sum().item()
            n_image_features = image_features.shape[0] * image_features.shape[1]

            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            special_image_mask = (
                (input_ids == self.config.image_token_index)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        logits = outputs[0][0]
        hidden_states = outputs[1]
        outputs = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1):].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        ), hidden_states

AutoModel.register(ResnetConfig, ResnetExpertModel)
# AutoModel.register(ResnetConfig, ResnetExpertModel)


# 示例使用
# config = LlavaConfig.from_pretrained("llava-base")
# model = CustomLlavaForConditionalGeneration(config)
# input_ids = torch.tensor([[1, 2, 3, 4]])
# attention_mask = torch.tensor([[1, 1, 1, 1]])
# output = model(input_ids, attention_mask=attention_mask)
# print(output)
# if __name__ == '__main__':
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

AutoModel.register(ResnetConfig, ResnetExpertModel)
AutoModel.register(LlavaWithVisionExpertConfig, CustomLlavaNextForConditionalGeneration)
if __name__ == '__main__':
    # pretrain_path = "/data/kesun/kesun/npr_backbone/experiment_name2025_01_03_16_14_28/model_epoch_0.8713537474688394_0.9277614465134129.pth"
    # resnet50d_config = ResnetConfig(pretrain_path=pretrain_path, num_classes=1, use_low_level="npr", pretrained=False)
    # resnet50d = ResnetExpertModel(resnet50d_config)
    #
    # # resnet50d_config.save_pretrained("custom-resnet-debug")
    #
    # # AutoConfig.register("resnet_expert", ResnetConfig)
    # # AutoModel.register(ResnetConfig, ResnetExpertModel)
    # b = torch.randn(1,3,224,224)
    # a = resnet50d(b)
    # print(a.shape)
    # resnet50d_2 = AutoModel.from_config(resnet50d_config)
    # aa = resnet50d_2(b)
    # print(aa.shape)

    from PIL import Image
    import requests
    from transformers import AutoProcessor, LlavaNextForConditionalGeneration, AutoConfig
    from swift.new_models.modeling_llava import CustomLlavaNextForConditionalGeneration
    from swift.new_models.config_llava_new import LlavaWithVisionExpertConfig, ResnetConfig
    # from xtuner.model.config_llava_new import LlavaWithVisionExpertConfig, ResnetConfig
    from swift.new_models.processing_llava_next import CustomLlavaNextProcessor

    resnet50_config = ResnetConfig.from_pretrained("/data/kesun/kesun/resnet-expert")

    from transformers import AutoImageProcessor, AutoModel
    from PIL import Image
    import requests

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    processor = AutoImageProcessor.from_pretrained('/data/kesun/kesun/facebook/dinov2-large')
    model1 = AutoModel.from_pretrained('/data/kesun/kesun/facebook/dinov2-large')

    inputs = processor(images=image, return_tensors="pt")
    # outputs = model(**inputs)
    # last_hidden_states = outputs.last_hidden_state

    # model1 = AutoModel.from_pretrained("/data/kesun/work_dirs/llava_v15_13b_finetune_AIGC_lora_best1128/llava165-7b-instruct/checkpoint-220-merged")
    # vision_expert = AutoModel.from_config(resnet50_config)
    # pretrain_path = "/data/kesun/kesun/npr_backbone/experiment_name2025_01_03_16_14_28/model_epoch_0.8713537474688394_0.9277614465134129.pth"
    config1 = AutoConfig.from_pretrained("/data/kesun/kesun/llava-v1.6-mistral-7b-hf")
    config = LlavaWithVisionExpertConfig(expert_config=model1.config, **config1.to_dict())
    config.vision_feature_select_strategy = "expert"
    model = CustomLlavaNextForConditionalGeneration.from_pretrained("/data/kesun/kesun/llava-v1.6-mistral-7b-hf",
                                                                    config=config)
    # processor = AutoProcessor.from_pretrained("/data/kesun/kesun/llava-v1.6-mistral-7b-hf-ours")
    processor = CustomLlavaNextProcessor.from_pretrained("/data/kesun/kesun/llava-v1.6-mistral-7b-hf")
    # model.vision_expert.from_pretrained("/data/kesun/kesun/resnet-expert", config=resnet50_config)
    # model1 = AutoModel.from_pretrained("/data/kesun/kesun/resnet-expert")
    model.vision_expert = model1
    save_path = "/data/kesun/kesun/llava-v1.6-mistral-7b-hf-with-expert-dino"
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
