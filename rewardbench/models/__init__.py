# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LlamaTokenizer,
    MixtralForCausalLM,
    T5ForConditionalGeneration,
    pipeline,
)

from .beaver import BeaverCostPipeline, BeaverPipeline, LlamaForScore
from .betterpairrm import BetterPairRMPipeline
from .openassistant import *  # noqa
from .openbmb import LlamaRewardModel, OpenBMBPipeline
from .pairrm import DebertaV2PairRM, PairRMPipeline
from .shp import SHPPipeline
from .starling import (
    LlamaForSequenceClassification,
    StarlingPipeline,
    build_starling_rm,
)
from .ziya import ZiyaPipeline

# Please open a PR if you need to add more custom modeling code / utilize existing code for you model
REWARD_MODEL_CONFIG = {
    "default": {
        "model_builder": AutoModelForSequenceClassification.from_pretrained,
        "pipeline_builder": pipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "berkeley-nest/Starling-RM-7B-alpha": {
        "model_builder": build_starling_rm,
        "pipeline_builder": StarlingPipeline,
        "quantized": False,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "Nexusflow/Starling-RM-34B": {
        "model_builder": LlamaForSequenceClassification.from_pretrained,
        "pipeline_builder": StarlingPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "openbmb/UltraRM-13b": {
        "model_builder": LlamaRewardModel.from_pretrained,
        "pipeline_builder": OpenBMBPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "openbmb/Eurus-RM-7b": {
        "model_builder": AutoModel.from_pretrained,
        "pipeline_builder": OpenBMBPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "llm-blender/PairRM-hf": {
        "model_builder": DebertaV2PairRM.from_pretrained,
        "pipeline_builder": PairRMPipeline,
        "quantized": True,
        "custom_dialogue": True,
        "model_type": "Custom Classifier",
    },
    "mightbe/Better-PairRM": {
        "model_builder": DebertaV2PairRM.from_pretrained,
        "pipeline_builder": BetterPairRMPipeline,
        "quantized": True,
        "custom_dialogue": True,
        "model_type": "Custom Classifier",
    },
    "stanfordnlp/SteamSHP-flan-t5-xl": {
        "model_builder": T5ForConditionalGeneration.from_pretrained,
        "pipeline_builder": SHPPipeline,
        "quantized": True,
        "custom_dialogue": True,
        "model_type": "Custom Classifier",
    },
    "stanfordnlp/SteamSHP-flan-t5-large": {
        "model_builder": T5ForConditionalGeneration.from_pretrained,
        "pipeline_builder": SHPPipeline,
        "quantized": True,
        "custom_dialogue": True,
        "model_type": "Custom Classifier",
    },
    "IDEA-CCNL/Ziya-LLaMA-7B-Reward": {
        "model_builder": AutoModelForSequenceClassification.from_pretrained,
        "pipeline_builder": ZiyaPipeline,
        "quantized": False,  # handled by .half() in the custom pipeline, as in model card
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "PKU-Alignment/beaver-7b-v1.0-reward": {
        "model_builder": LlamaForScore.from_pretrained,
        "pipeline_builder": BeaverPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "PKU-Alignment/beaver-7b-v1.0-cost": {
        "model_builder": LlamaForScore.from_pretrained,
        "pipeline_builder": BeaverCostPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
}

DPO_MODEL_CONFIG = {
    "default": {
        "model_builder": AutoModelForCausalLM.from_pretrained,
        "tokenizer_builder": AutoTokenizer.from_pretrained,
    },
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": {
        "model_builder": MixtralForCausalLM.from_pretrained,
        "tokenizer_builder": LlamaTokenizer.from_pretrained,
    },
}
