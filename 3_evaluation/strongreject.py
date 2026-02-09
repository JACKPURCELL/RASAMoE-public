"""
StrongReject dataset handler.
"""

from typing import Dict, Any
from datasets import Dataset, load_dataset
from base import BaseDatasetHandler
from judge import evaluate_with_judge


class StrongRejectHandler(BaseDatasetHandler):
    """Handler for StrongReject safety evaluation dataset."""
    
    def get_dataset_name(self) -> str:
        return "strongreject"
    
    def get_evaluation_type(self) -> str:
        return "safety"
    
    def load_dataset(self) -> Dataset:
        """Load StrongReject dataset from HuggingFace."""
        strong_reject_dataset = load_dataset("walledai/StrongReject", split="train")
        remove_cols = strong_reject_dataset.column_names
        
        def prepare_strong(example):
            return {"prompt": example["prompt"], "dataset_name": "StrongReject"}

        return strong_reject_dataset.map(prepare_strong, remove_columns=remove_cols)
    
    def prepare_prompt(self, sample: Dict[str, Any]) -> str:
        """Prepare prompt for StrongReject safety evaluation."""
        return sample["prompt"]
    
    def evaluate_response(self, prompt: str, response: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate response using GPT judge for safety compliance."""
        return evaluate_with_judge(prompt, response)
    
    def calculate_metrics(self, results: list) -> Dict[str, float]:
        """Calculate Safety Rate: count of score=0 (safe responses) / total samples."""
        if not results:
            return {"safety_rate": 0.0}
        
        safe_count = sum(1 for result in results if result.get("evaluation", {}).get("score", 1) == 0)
        safety_rate = safe_count / len(results)
        
        print(f"StrongReject Safety Rate: {safety_rate:.4f} ({safe_count}/{len(results)} safe responses)")
        return {"safety_rate": safety_rate}