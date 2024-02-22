from transformers import (
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration,
    pipeline,
)

from .beaver import BeaverPipeline, LlamaForScore
from .openassistant import *  # noqa
from .openbmb import LlamaRewardModel, OpenBMBPipeline
from .pairrm import DebertaV2PairRM, PairRMPipeline
from .shp import SHPPipeline
from .starling import StarlingPipeline, build_starling_rm
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
    "berkeley-nest/Starling-RM-34B": {
        "model_builder": build_starling_rm,
        "pipeline_builder": StarlingPipeline,
        "quantized": False,
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
    "llm-blender/PairRM-hf": {
        "model_builder": DebertaV2PairRM.from_pretrained,
        "pipeline_builder": PairRMPipeline,
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
        "pipeline_builder": BeaverPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
}
