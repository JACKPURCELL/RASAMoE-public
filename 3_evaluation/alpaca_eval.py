"""
Alpaca Eval dataset handler.
"""

from typing import Dict, Any
from datasets import Dataset, load_dataset
from base import BaseDatasetHandler
import json

class AlpacaEvalHandler(BaseDatasetHandler):
    """Handler for Alpaca Eval dataset."""
    
    def get_dataset_name(self) -> str:
        return "alpaca_eval"
    
    def get_evaluation_type(self) -> str:
        return "generation"
    
    def load_dataset(self) -> Dataset:
        """Load Alpaca Eval dataset from local JSON file."""
        # Read JSON file directly
        with open('./3_evaluation/alpaca_eval.json', 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} samples from alpaca_eval.json")
        
        # Prepare data for Dataset creation
        prepared_data = []
        for example in data:
            prepared_example = {
                "prompt": example["instruction"], 
                "dataset_name": "alpaca_eval",
                "instruction": example["instruction"],  # Keep original instruction
                "reference_output": example.get("output", ""),  # Keep reference output if available
                "original_generator": example.get("generator", ""),  # Keep original generator
                "dataset_source": example.get("dataset", ""),  # Keep dataset source
                "example_id": example.get("id", "")  # Keep ID if available
            }
            prepared_data.append(prepared_example)
        
        # Create Dataset from the prepared data
        from datasets import Dataset
        return Dataset.from_list(prepared_data)
    
    def prepare_prompt(self, sample: Dict[str, Any]) -> str:
        """Prepare prompt for Alpaca Eval instruction following."""
        return sample['instruction']
    
    def evaluate_response(self, prompt: str, response: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        For Alpaca Eval, we don't calculate scores. 
        Just return the response in the expected format for JSON output.
        """
        return {
            "instruction": sample.get("instruction", prompt),
            "output": response,
            "generator": "my_model",  # As requested in the user's code
            "reference_output": sample.get("reference_output", ""),
            "original_generator": sample.get("original_generator", ""),  # Keep track of original generator
            "dataset_source": sample.get("dataset_source", ""),  # Keep track of dataset source
            "example_id": sample.get("example_id", ""),
            "score": None,  # No scoring for this dataset
            "skip": False  # Don't skip any samples
        }
    
    def calculate_metrics(self, results: list) -> Dict[str, float]:
        """
        Calculate metrics for Alpaca Eval.
        Since we don't score responses, just return count information.
        """
        if not results:
            return {"total_samples": 0, "processed_samples": 0}
        
        # Count processed samples (non-skipped)
        processed_count = sum(1 for result in results 
                            if not result.get("evaluation", {}).get("skip", False))
        
        return {
            "total_samples": len(results),
            "processed_samples": processed_count,
            "completion_rate": processed_count / len(results) if results else 0.0
        }
