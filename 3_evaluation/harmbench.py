"""
HarmBench dataset handler.
"""

from typing import Dict, Any
from datasets import Dataset, load_dataset
from base import BaseDatasetHandler
from judge import evaluate_with_judge


class HarmBenchHandler(BaseDatasetHandler):
    """Handler for HarmBench safety evaluation dataset."""
    
    def get_dataset_name(self) -> str:
        return "harmbench"
    
    def get_evaluation_type(self) -> str:
        return "safety"
    
    def load_dataset(self) -> Dataset:
        """Load HarmBench dataset from HuggingFace."""
        harmbench_dataset = load_dataset("walledai/HarmBench", 'standard', split="train")
        remove_cols = harmbench_dataset.column_names
        
        def prepare_harmbench(example):
            return {"prompt": example["prompt"], "dataset_name": "HarmBench"}

        return harmbench_dataset.map(prepare_harmbench, remove_columns=remove_cols)
    
    def prepare_prompt(self, sample: Dict[str, Any]) -> str:
        """Prepare prompt for HarmBench safety evaluation."""
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
        
        print(f"HarmBench Safety Rate: {safety_rate:.4f} ({safe_count}/{len(results)} safe responses)")
        return {"safety_rate": safety_rate}