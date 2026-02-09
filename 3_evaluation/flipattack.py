"""
FLIPATTACK dataset handler for jailbreak attack evaluation.
"""

import json
from typing import Dict, Any, Optional
from datasets import Dataset
from base import BaseDatasetHandler
from judge import evaluate_with_judge


class FLIPATTACKHandler(BaseDatasetHandler):
    """Handler for FLIPATTACK jailbreak attack evaluation dataset."""
    
    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize FLIPATTACK handler with custom dataset path.
        
        Args:
            dataset_path: Path to the FLIPATTACK.json file.
                         Defaults to ~/RASAMoE/FLIPATTACK.json
        """
        self.dataset_path = dataset_path or "~/RASAMoE/FLIPATTACK.json"
        super().__init__()
    
    def get_dataset_name(self) -> str:
        return "flipattack"
    
    def get_evaluation_type(self) -> str:
        return "safety"
    
    def load_dataset(self) -> Dataset:
        """Load FLIPATTACK dataset from JSON file."""
        with open(self.dataset_path, 'r') as f:
            dataset = json.load(f)
        
        # Process dataset items
        processed_items = []
        for item in dataset:
            # Skip items that were not successfully judged by GPT-4
            if item.get("judge_success_gpt4", 0) == 0:
                continue
            
            goal = item["goal"]
            unsafe_messages = item["all_prompt"]
            
            # Store both the original goal and unsafe messages
            processed_items.append({
                "goal": goal,
                "unsafe_messages": unsafe_messages,
                "dataset_name": "FLIPATTACK"
            })
        
        # Convert to HuggingFace Dataset format
        return Dataset.from_list(processed_items)
    
    def prepare_prompt(self, sample: Dict[str, Any]) -> Any:
        """
        Prepare prompt for FLIPATTACK evaluation.
        Returns the unsafe_messages (complete jailbreak attack prompt).
        """
        # Return the unsafe messages as is (it's already a list of message dicts)
        return sample["unsafe_messages"]
    
    def evaluate_response(self, prompt: Any, response: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate response using GPT judge for safety compliance.
        Uses the original goal (not the jailbreak prompt) for evaluation context.
        
        Args:
            prompt: The unsafe_messages that were used for generation
            response: The model's response to the unsafe_messages
            sample: The original sample containing the goal
            
        Returns:
            Evaluation result from judge with score and reason
        """
        # Use the original goal for judge evaluation, not the jailbreak prompt
        original_goal = sample["goal"]
        return evaluate_with_judge(original_goal, response)
    
    def calculate_metrics(self, results: list) -> Dict[str, float]:
        """
        Calculate Safety Rate: count of score=0 (safe responses) / total samples.
        Same metric as HarmBench.
        """
        if not results:
            return {"safety_rate": 0.0}
        
        safe_count = sum(1 for result in results if result.get("evaluation", {}).get("score", 1) == 0)
        safety_rate = safe_count / len(results)
        
        print(f"FLIPATTACK Safety Rate: {safety_rate:.4f} ({safe_count}/{len(results)} safe responses)")
        return {"safety_rate": safety_rate}

