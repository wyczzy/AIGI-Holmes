from transformers import LlavaConfig, LlavaNextConfig
from transformers import PretrainedConfig
from typing import List
from transformers import AutoConfig
from transformers.models.auto import CONFIG_MAPPING
from transformers.configuration_utils import PretrainedConfig

class LlavaWithResNetConfig(LlavaConfig):
    model_type = "llava_with_resnet"  # 模型类型名称，用于标识模型

    def __init__(self, resnet_hidden_size=2048, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resnet_hidden_size = resnet_hidden_size


# class LlavaWithVisionExpertConfig(LlavaNextConfig):
#     model_type = "llava_with_vision_expert"
#
#     def __init__(self, expert_config=None, *args, **kwargs):
#         # 保存原始策略值
#         original_strategy1 = kwargs.get("vision_feature_select_strategy", None)
#         original_strategy2 = args.get("vision_feature_select_strategy", None)
#         if original_strategy1 == None and original_strategy2 == None:
#             raise ValueError(
#                     "vision_feature_select_strategy must be 'default', 'full', or 'expert'. "
#                     f"Got: {original_strategy1}, {original_strategy2}"
#                 )
#
#         # # 验证是否包含新策略
#         # if original_strategy not in ["default", "full", "expert"]:
#         #     raise ValueError(
#         #         "vision_feature_select_strategy must be 'default', 'full', or 'expert'. "
#         #         f"Got: {original_strategy}"
#         #     )
#
#         # 临时替换策略以通过父类验证
#         if original_strategy1 == "expert":
#             # args["vision_feature_select_strategy"] = "default"
#             kwargs["vision_feature_select_strategy"] = "default"  # 使用父类允许的值
#         elif original_strategy2 == "expert":
#             args["vision_feature_select_strategy"] = "default"
#         # 调用父类初始化
#         super().__init__(*args, **kwargs)
#
#         # 恢复原始策略值
#         if original_strategy1 != None:
#             self.vision_feature_select_strategy = original_strategy1
#         elif original_strategy2 != None:
#             self.vision_feature_select_strategy = original_strategy2
#
#         # 处理专家配置
#         if isinstance(expert_config, dict):
#             expert_config["model_type"] = (
#                 expert_config["model_type"] if "model_type" in expert_config else "resnet_expert"
#             )
#             expert_config = CONFIG_MAPPING[expert_config["model_type"]](**expert_config)
#         self.expert_config = expert_config

# 这个方案需要对/home/kesun/anaconda3/envs/llava/lib/python3.10/site-packages/transformers/models/llava_next/configuration_llava_next.py
# 进行修改
class LlavaWithVisionExpertConfig(LlavaNextConfig):
    model_type = "llava_with_vision_expert"


    def __init__(self, expert_config=None, *args, **kwargs):
        if isinstance(expert_config, dict):
            expert_config["model_type"] = (
                expert_config["model_type"] if "model_type" in expert_config else "resnet_expert"
            )
            expert_config = CONFIG_MAPPING[expert_config["model_type"]](**expert_config)
        self.expert_config = expert_config
        super().__init__(*args, **kwargs)
    # is_composition = False
    #
    # def __init__(
    #         self,
    #         expert_config=None,
    #         vision_config=None,
    #         text_config=None,
    #         ignore_index=-100,
    #         image_token_index=32000,
    #         projector_hidden_act="gelu",
    #         vision_feature_select_strategy="default",
    #         vision_feature_layer=-2,
    #         image_grid_pinpoints=None,
    #         tie_word_embeddings=False,
    #         image_seq_length=576,
    #         **kwargs,
    # ):
    #     if isinstance(expert_config, dict):
    #         expert_config["model_type"] = (
    #                 expert_config["model_type"] if "model_type" in expert_config else "resnet_expert"
    #             )
    #         expert_config = CONFIG_MAPPING[expert_config["model_type"]](**expert_config)
    #     self.expert_config = expert_config
    #     self.ignore_index = ignore_index
    #     self.image_token_index = image_token_index
    #     self.projector_hidden_act = projector_hidden_act
    #     self.image_seq_length = image_seq_length
    #
    #     if vision_feature_select_strategy not in ["default", "full", "expert"]:
    #         raise ValueError(
    #             "vision_feature_select_strategy should be one of 'default', 'full', 'expert'."
    #             f"Got: {vision_feature_select_strategy}"
    #         )
    #
    #     self.vision_feature_select_strategy = vision_feature_select_strategy
    #     self.vision_feature_layer = vision_feature_layer
    #     image_grid_pinpoints = (
    #         image_grid_pinpoints
    #         if image_grid_pinpoints is not None
    #         else [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
    #     )
    #     self.image_grid_pinpoints = image_grid_pinpoints
    #
    #     if isinstance(vision_config, dict):
    #         vision_config["model_type"] = (
    #             vision_config["model_type"] if "model_type" in vision_config else "clip_vision_model"
    #         )
    #         vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
    #     elif vision_config is None:
    #         vision_config = CONFIG_MAPPING["clip_vision_model"](
    #             intermediate_size=4096,
    #             hidden_size=1024,
    #             patch_size=14,
    #             image_size=336,
    #             num_hidden_layers=24,
    #             num_attention_heads=16,
    #             vocab_size=32000,
    #             projection_dim=768,
    #         )
    #
    #     self.vision_config = vision_config
    #
    #     if isinstance(text_config, dict):
    #         text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
    #         text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
    #     elif text_config is None:
    #         text_config = CONFIG_MAPPING["llama"]()
    #
    #     self.text_config = text_config
    #
    #     super().__init__(tie_word_embeddings=tie_word_embeddings)

from transformers import PretrainedConfig
from typing import List


class ResnetConfig(PretrainedConfig):
    model_type = "resnet_expert"

    def __init__(
        self,
        num_classes: int = 1,
        pretrain_path: str = "",
        use_low_level: str = "npr",
        pretrained: bool = False,
        **kwargs,
    ):

        self.num_classes = num_classes
        self.pretrain_path = pretrain_path
        self.use_low_level = use_low_level
        self.pretrained = pretrained
        super().__init__(**kwargs)




AutoConfig.register("resnet_expert", ResnetConfig)
AutoConfig.register("llava_with_vision_expert", LlavaWithVisionExpertConfig)