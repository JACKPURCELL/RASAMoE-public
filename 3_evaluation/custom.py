"""
Custom dataset handler.
"""

import json
import os
from typing import Dict, Any
from datasets import Dataset
from base import BaseDatasetHandler
from judge import evaluate_with_judge


class CustomHandler(BaseDatasetHandler):
    """Handler for custom vulnerability report dataset."""
    
    def __init__(self, vulnerability_report_path: str = None):
        self.vulnerability_report_path = vulnerability_report_path
        super().__init__()
    
    def get_dataset_name(self) -> str:
        return "custom"
    
    def get_evaluation_type(self) -> str:
        return "safety"
    
    def load_dataset(self) -> Dataset:
        """Load custom dataset from vulnerability report JSON file."""
        if not self.vulnerability_report_path or not os.path.exists(self.vulnerability_report_path):
            raise FileNotFoundError(f"Vulnerability report file not found: {self.vulnerability_report_path}")
            
        with open(self.vulnerability_report_path, "r") as f:
            my_raw_data_list = json.load(f)

        keep_data_list = [
            data for data in my_raw_data_list
            if "Type B: Core LLM Failure" in data["failure_types"] or "Type C: Systemic Failure" in data["failure_types"]
        ]
        
        custom_dataset = Dataset.from_list(keep_data_list)
        split_dataset = custom_dataset.train_test_split(test_size=0.1, seed=42)
        eval_custom = split_dataset['test']
        
        def prepare_custom(example):
            return {"prompt": example["adversarial_prompt"], "dataset_name": "custom"}

        return eval_custom.map(prepare_custom, remove_columns=eval_custom.column_names)
    
    def prepare_prompt(self, sample: Dict[str, Any]) -> str:
        """Prepare prompt for custom safety evaluation."""
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