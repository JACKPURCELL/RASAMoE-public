"""
GSM8K dataset handler.
"""

import re
from typing import Dict, Any
from datasets import Dataset, load_dataset
from base import BaseDatasetHandler


class GSM8KHandler(BaseDatasetHandler):
    """Handler for GSM8K (Grade School Math 8K) dataset."""
    
    def get_dataset_name(self) -> str:
        return "gsm8k"
    
    def get_evaluation_type(self) -> str:
        return "math_problem"
    
    def load_dataset(self) -> Dataset:
        """Load GSM8K dataset from HuggingFace."""
        gsm8k_dataset = load_dataset("openai/gsm8k", "main", split="test")
        remove_cols = gsm8k_dataset.column_names

        def prepare_gsm8k(example):
            return {
                "prompt": example["question"], 
                "dataset_name": "gsm8k",
                "answer": example["answer"]
            }

        return gsm8k_dataset.map(prepare_gsm8k, remove_columns=remove_cols)
    
    def prepare_prompt(self, sample: Dict[str, Any]) -> str:
        """Prepare prompt for GSM8K math problem."""
        return f"{sample['prompt']}\n\nPlease solve this step by step and provide your final numerical answer within \\boxed{{}}."
    
    def evaluate_response(self, prompt: str, response: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate GSM8K response by comparing numerical answers."""
        # Extract numerical answer from the correct answer string (after ####)
        answer_match = re.search(r'####\s*([+-]?\d+(?:\.\d+)?)', sample["answer"])
        if answer_match:
            correct_numerical = float(answer_match.group(1))
        else:
            print(f"❌ Could not extract numerical answer from: {sample['answer']}")
            return {
                "predicted_answer": None,
                "correct_answer": sample["answer"],
                "is_correct": False,
                "score": 0
            }
        
        # Extract answer from response - look for \boxed{} pattern
        response_clean = response.strip()
        
        # Try different patterns to extract the numerical answer from \boxed{}
        patterns = [
            r'\\boxed\{([+-]?\d+(?:\.\d+)?)\}',  # \boxed{123} or \boxed{12.5}
            r'boxed\{([+-]?\d+(?:\.\d+)?)\}',    # boxed{123} (without backslash)
            r'\\boxed\{([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\}',  # \boxed{1,234} with commas
            r'boxed\{([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\}',    # boxed{1,234} with commas
        ]
        
        predicted_numerical = None
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, response_clean)
            if match:
                # Remove commas and convert to float
                number_str = match.group(1).replace(',', '')
                try:
                    predicted_numerical = float(number_str)
                    print(f"✅ Pattern {i+1} matched: Found numerical answer '{predicted_numerical}' in: '{match.group(0)}'")
                    break
                except ValueError:
                    continue
        
        # If no boxed pattern found, try to find the last number in the response
        if predicted_numerical is None:
            # Look for the last standalone number in the response
            number_matches = re.findall(r'([+-]?\d+(?:\.\d+)?)', response_clean)
            if number_matches:
                try:
                    predicted_numerical = float(number_matches[-1])
                    print(f"🔧 Fallback: Using last number '{predicted_numerical}' from response")
                except ValueError:
                    pass
        
        # Default to 0 if nothing found
        if predicted_numerical is None:
            predicted_numerical = 0
            print(f"❌ No numerical answer found in response")
        
        # Compare numerical values (allow small floating point differences)
        is_correct = abs(predicted_numerical - correct_numerical) < 1e-6
        
        return {
            "predicted_answer": predicted_numerical,
            "correct_answer": correct_numerical,
            "is_correct": is_correct,
            "score": 1 if is_correct else 0
        }
