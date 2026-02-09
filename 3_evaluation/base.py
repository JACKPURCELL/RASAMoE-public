"""
Base class for dataset handlers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datasets import Dataset


class BaseDatasetHandler(ABC):
    """Base class for all dataset handlers."""
    
    def __init__(self):
        self.dataset_name = self.get_dataset_name()
        self.dataset = None
        
    @abstractmethod
    def get_dataset_name(self) -> str:
        """Return the name of this dataset."""
        pass
    
    @abstractmethod
    def load_dataset(self) -> Dataset:
        """Load and prepare the dataset."""
        pass
    
    @abstractmethod
    def prepare_prompt(self, sample: Dict[str, Any]) -> str:
        """Prepare the prompt for a single sample."""
        pass
    
    @abstractmethod
    def evaluate_response(self, prompt: str, response: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the model response for a single sample."""
        pass
    
    def get_evaluation_type(self) -> str:
        """Return the type of evaluation (multiple_choice, math_problem, safety, etc.)."""
        return "default"
    
    def prepare_dataset(self) -> Dataset:
        """Load and prepare the dataset if not already loaded."""
        if self.dataset is None:
            print(f"Loading {self.dataset_name} dataset...")
            try:
                self.dataset = self.load_dataset()
                print(f"✅ {self.dataset_name} dataset loaded: {len(self.dataset)} samples")
            except Exception as e:
                print(f"⚠️ Warning: Failed to load {self.dataset_name} dataset: {e}")
                return None
        return self.dataset
    
    def select_samples(self, num_samples: int, seed: int = 42) -> Optional[Dataset]:
        """Select a subset of samples for evaluation."""
        dataset = self.prepare_dataset()
        if dataset is None:
            return None
        
        return dataset.shuffle(seed=seed).select(range(min(num_samples, len(dataset))))
    
    def calculate_metrics(self, results: list) -> Dict[str, float]:
        """Calculate dataset-specific metrics from evaluation results.
        
        Default implementation calculates average score, excluding skipped samples.
        Override in subclasses for specialized metrics.
        """
        if not results:
            return {"avg_score": 0.0, "total_samples": 0, "skipped_samples": 0}
        
        # Filter out skipped samples
        valid_results = []
        skipped_count = 0
        
        for result in results:
            evaluation = result.get("evaluation", {})
            if evaluation.get("skip", False):
                skipped_count += 1
            else:
                valid_results.append(result)
        
        if not valid_results:
            return {"avg_score": 0.0, "total_samples": len(results), "skipped_samples": skipped_count}
        
        total_score = sum(result.get("evaluation", {}).get("score", 0) for result in valid_results)
        avg_score = total_score / len(valid_results)
        
        return {
            "avg_score": avg_score,
            "total_samples": len(results),
            "valid_samples": len(valid_results),
            "skipped_samples": skipped_count
        }