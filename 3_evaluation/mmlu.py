"""
MMLU dataset handler.
"""

import re
from typing import Dict, Any
from datasets import Dataset, load_dataset
from base import BaseDatasetHandler


class MMLUHandler(BaseDatasetHandler):
    """Handler for MMLU (Massive Multitask Language Understanding) dataset."""
    
    def get_dataset_name(self) -> str:
        return "mmlu"
    
    def get_evaluation_type(self) -> str:
        return "multiple_choice"
    
    def load_dataset(self) -> Dataset:
        """Load MMLU dataset from HuggingFace."""
        mmlu_dataset = load_dataset("cais/mmlu", "all", split="test")
        remove_cols = mmlu_dataset.column_names

        def prepare_mmlu(example):
            # Convert numeric answer to letter (0->A, 1->B, 2->C, 3->D)
            answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
            choices_dict = {
                "label": ["A", "B", "C", "D"],
                "text": example["choices"]
            }
            return {
                "prompt": example["question"], 
                "dataset_name": "mmlu",
                "subject": example["subject"],
                "choices": choices_dict,
                "answerKey": answer_map[example["answer"]]
            }

        return mmlu_dataset.map(prepare_mmlu, remove_columns=remove_cols)
    
    def prepare_prompt(self, sample: Dict[str, Any]) -> str:
        """Prepare prompt for MMLU multiple choice question."""
        choices_text = "\n".join([
            f"{label}. {text}" 
            for label, text in zip(sample["choices"]["label"], sample["choices"]["text"])
        ])
        return f"{sample['prompt']}\n\nChoices:\n{choices_text}\n\n You can analyze first, then include the final answer choice(ONLY letter) within \\boxed{{}}. Answer:"
    
    def evaluate_response(self, prompt: str, response: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate MMLU response by extracting letter choice."""
        # Extract answer from response - look for A, B, C, D, E
        response_clean = response.strip().upper()
        
        # Determine valid choices based on available options
        valid_choices = sample["choices"]["label"] if isinstance(sample["choices"], dict) else ["A", "B", "C", "D", "E"]
        choice_pattern = "|".join(valid_choices)
        
        # Try different patterns to extract the answer (ordered by priority)
        patterns = [
            rf'(?i)\\boxed\{{({choice_pattern})\}}',  # \boxed{A} - highest priority, case insensitive
            rf'(?i)boxed\{{({choice_pattern})\}}',    # boxed{A} (without backslash), case insensitive
            rf'(?i)\\\(\s*\\boxed\{{({choice_pattern})\}}\s*\\\)',  # \(\boxed{A}\), case insensitive
            rf'(?:THE\s+)?(?:ANSWER|FINAL ANSWER|CHOICE|OPTION)\s*(?:IS\s*)?({choice_pattern})\b',  # "answer is A"
            rf'(?:CLOSEST|IS)\s+(?:ANSWER\s+)?(?:CHOICE\s+)?(?:TO\s+)?({choice_pattern})\b',  # "closest answer choice A"
            rf'^({choice_pattern})\b',  # Answer at very start of response
            rf'\b({choice_pattern})\b(?=\s*[.!?]*\s*$)',  # Answer at very end with punctuation
            rf'\(({choice_pattern})\)(?=\s*[.!?]*\s*$)',  # (A) at end only
        ]
        
        predicted_answer = None
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, response_clean)
            if match:
                predicted_answer = match.group(1)
                print(f"✅ Pattern {i+1} matched: '{pattern}' -> Found: '{predicted_answer}' in: '{match.group(0)}'")
                break
        
        if predicted_answer is None:
            print(f"❌ No pattern matched. Response (last 200 chars): ...{response_clean[-200:]}")
            # As a fallback, try a more lenient search for any boxed pattern
            fallback_match = re.search(rf'boxed\{{({choice_pattern})\}}', response_clean, re.IGNORECASE)
            if fallback_match:
                predicted_answer = fallback_match.group(1)
                print(f"🔧 Fallback matched: Found '{predicted_answer}'")
        
        # If no clear answer found, try to match with choice text
        if predicted_answer is None:
            choice_texts = [choice.lower() for choice in sample["choices"]["text"]]
            response_lower = response.lower()
            
            for i, choice_text in enumerate(choice_texts):
                if choice_text in response_lower:
                    predicted_answer = sample["choices"]["label"][i]
                    break
        
        # If still no answer found, mark this sample to be skipped
        if predicted_answer is None:
            print(f"❌ No valid answer found - skipping this sample")
            return {
                "predicted_answer": None,
                "correct_answer": sample["answerKey"].upper(),
                "is_correct": False,
                "score": 0,
                "subject": sample.get("subject", "unknown"),
                "skip": True  # Mark this sample to be skipped in metric calculation
            }
        
        correct_answer = sample["answerKey"].upper()
        is_correct = predicted_answer == correct_answer
        
        return {
            "predicted_answer": predicted_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "score": 1 if is_correct else 0,
            "subject": sample.get("subject", "unknown"),
            "skip": False
        }
