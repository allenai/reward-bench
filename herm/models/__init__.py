from transformers import (
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration,
    pipeline,
)

from .openbmb import LlamaRewardModel, OpenBMBPipeline
from .pairrm import DebertaV2PairRM, PairRMPipeline
from .shp import SHPPipeline
from .starling import StarlingPipeline, build_starling_rm

__all__ = [
    "LlamaRewardModel",
    "OpenBMBPipeline",
    "DebertaV2PairRM",
    "PairRMPipeline",
    "SHPPipeline",
    "StarlingPipeline",
    "build_starling_rm",
]

REWARD_MODEL_CONFIG = {
    "default": {
        "model_builder": AutoModelForSequenceClassification.from_pretrained,
        "pipeline_builder": pipeline,
        "quantized": True,
        "custom_dialogue": False,
    },
    "oasst": {
        "model_builder": AutoModelForSequenceClassification.from_pretrained,
        "pipeline_builder": pipeline,
        "quantized": True,
        "custom_dialogue": False,
        "models": [
            "OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1",
            "OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5",
            "OpenAssistant/reward-model-deberta-v3-base",
            "OpenAssistant/reward-model-deberta-v3-large",
            "OpenAssistant/reward-model-deberta-v3-large-v2",
            "OpenAssistant/reward-model-electra-large-discriminator",
        ],
    },
    "Starling": {
        "model_builder": build_starling_rm,
        "pipeline_builder": StarlingPipeline,
        "quantized": False,
        "custom_dialogue": False,
        "models": [
            "berkeley-nest/Starling-RM-7B-alpha",
        ],
    },
    "openbmb": {
        "model_builder": LlamaRewardModel.from_pretrained,
        "pipeline_builder": OpenBMBPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "models": ["openbmb/UltraRM-13b"],
    },
    "PairRM": {
        "model_builder": DebertaV2PairRM.from_pretrained,
        "pipeline_builder": PairRMPipeline,
        "quantized": True,
        "custom_dialogue": True,
        "models": [
            "llm-blender/PairRM",
            "llm-blender/PairRM-hf",
        ],
    },
    "SHP": {
        "model_builder": T5ForConditionalGeneration.from_pretrained,
        "pipeline_builder": SHPPipeline,
        "quantized": True,
        "custom_dialogue": True,
        "models": [
            "stanfordnlp/SteamSHP-flan-t5-large",
            "stanfordnlp/SteamSHP-flan-t5-xl",
        ],
    },
}
