"""
Dataset handlers for model evaluation.
"""

from .base import BaseDatasetHandler
from .mmlu import MMLUHandler
from .gsm8k import GSM8KHandler
from .strongreject import StrongRejectHandler
from .harmbench import HarmBenchHandler
from .pku_safe import PKUSafeHandler
from .custom import CustomHandler
from .truthfulqa import TruthfulQAHandler
from .xstest import XSTestHandler
from .alpaca_eval import AlpacaEvalHandler

# Registry of all available dataset handlers
DATASET_HANDLERS = {
    "mmlu": MMLUHandler,
    "gsm8k": GSM8KHandler,
    "strongreject": StrongRejectHandler,
    "harmbench": HarmBenchHandler,
    "pku_safe": PKUSafeHandler,
    "custom": CustomHandler,
    "truthfulqa": TruthfulQAHandler,
    "xstest": XSTestHandler,
    "alpaca_eval": AlpacaEvalHandler,
}

__all__ = [
    "BaseDatasetHandler",
    "MMLUHandler", 
    "GSM8KHandler",
    "StrongRejectHandler",
    "HarmBenchHandler", 
    "PKUSafeHandler",
    "CustomHandler",
    "TruthfulQAHandler",
    "XSTestHandler",
    "AlpacaEvalHandler",
    "DATASET_HANDLERS"
]
