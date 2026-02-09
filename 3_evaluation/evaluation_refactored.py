"""
Refactored model evaluation system with modular dataset handlers.
"""

import json
import os
import torch
from typing import List, Dict, Any, Optional
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from base import BaseDatasetHandler
from mmlu import MMLUHandler
from gsm8k import GSM8KHandler
from strongreject import StrongRejectHandler
from harmbench import HarmBenchHandler
from pku_safe import PKUSafeHandler
from custom import CustomHandler
from truthfulqa import TruthfulQAHandler
from xstest import XSTestHandler
from alpaca_eval import AlpacaEvalHandler
from flipattack import FLIPATTACKHandler
from deepinception import DEEPINCEPTIONPlusHandler
from johnny import JOHNNYHandler
from xteaming import XteamingHandler
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
    "flipattack": FLIPATTACKHandler,
    "johnny": JOHNNYHandler,
    "xteaming": XteamingHandler,
}


class ModelEvaluator:
    """Model evaluator with modular dataset handlers."""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-3B-Instruct",
                 output_dir: str = None,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 vllm: bool = False,
                 prefix: str = "RAW",
                 vulnerability_report_path: str = None,
                 flipattack_path: str = None,
                 deepinception_path: str = None,
                 johnny_path: str = None,
                 xteaming_path: str = None,
                 tensor_parallel_size: int = 1):
        """
        Initialize the model evaluator.
        
        Args:
            model_name: Path or name of the model to evaluate
            output_dir: Directory to save evaluation results
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            vllm: Whether to use vLLM for inference
            prefix: Prefix for output files
            vulnerability_report_path: Path to custom vulnerability report JSON
            flipattack_path: Path to FLIPATTACK dataset JSON file
            deepinception_path: Path to DEEPINCEPTION dataset JSON file
            johnny_path: Path to Johnny dataset JSON file
            xteaming_path: Path to Xteaming dataset JSON file
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.temperature = temperature
        self.top_p = top_p
        self.vllm = vllm
        self.prefix = prefix
        self.vulnerability_report_path = vulnerability_report_path
        self.flipattack_path = flipattack_path
        self.deepinception_path = deepinception_path    
        self.johnny_path = johnny_path
        self.xteaming_path = xteaming_path
        self.tensor_parallel_size = tensor_parallel_size
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True, 
            padding_side='left'
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model
        print(f"Initializing LLM from: {self.model_name}")
        if self.vllm:
            self.llm = LLM(model=self.model_name, tensor_parallel_size=self.tensor_parallel_size, gpu_memory_utilization=0.95)
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                trust_remote_code=True, 
                device_map="auto"
            )
        
        # Dataset handlers
        self.dataset_handlers = {}
    
    def load_datasets(self, dataset_names: List[str]) -> Dict[str, Any]:
        """
        Load specified datasets using their handlers.
        
        Args:
            dataset_names: List of dataset names to load
            
        Returns:
            Dictionary mapping dataset names to their handlers
        """
        self.dataset_handlers = {}
        
        for dataset_name in dataset_names:
            print(f"Loading dataset: {dataset_name}")
            
            if dataset_name == "custom":
                # Custom dataset requires special initialization
                handler = CustomHandler(self.vulnerability_report_path)
            elif dataset_name == "flipattack":
                # FLIPATTACK dataset requires special initialization
                handler = FLIPATTACKHandler(self.flipattack_path)
            elif dataset_name == "deepinception":
                handler = DEEPINCEPTIONPlusHandler(self.deepinception_path)
            elif dataset_name == "johnny":
                handler = JOHNNYHandler(self.johnny_path)
            elif dataset_name == "xteaming":
                handler = XteamingHandler(self.xteaming_path)
            elif dataset_name in DATASET_HANDLERS:
                handler = DATASET_HANDLERS[dataset_name]()
            else:
                print(f"⚠️ Warning: Unknown dataset '{dataset_name}'. Skipping.")
                continue
            
            # Load the dataset
            try:
                dataset = handler.prepare_dataset()
                if dataset is not None:
                    self.dataset_handlers[dataset_name] = handler
                    print(f"✅ {dataset_name} loaded successfully")
                else:
                    print(f"❌ Failed to load {dataset_name}")
            except Exception as e:
                print(f"⚠️ Warning: Error loading {dataset_name}: {e}")
        
        total_samples = sum(
            len(handler.prepare_dataset()) 
            for handler in self.dataset_handlers.values() 
            if handler.prepare_dataset() is not None
        )
        print(f"Total datasets loaded: {len(self.dataset_handlers)}")
        print(f"Total evaluation samples: {total_samples}")
        
        return self.dataset_handlers
    
    def generate_responses(self, prompts: List[Any], batch_size: int = 8) -> List[str]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts: List of prompts (either strings or message lists) to generate responses for
            batch_size: Batch size for generation
            
        Returns:
            List of generated responses
        """
        print(f"Generating responses in batches of {batch_size}...")
        all_responses = []
        
        # Batch generation
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
            
            # Prepare batch messages
            # Check if prompts are already message lists or simple strings
            batch_messages = []
            for prompt in batch_prompts:
                if isinstance(prompt, list):
                    # Already a list of messages (e.g., from FLIPATTACK)
                    batch_messages.append(prompt)
                else:
                    # Simple string prompt, wrap it in user message
                    batch_messages.append([{"role": "user", "content": prompt}])
            
            # Apply chat template
            batch_texts = self.tokenizer.apply_chat_template(
                batch_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            
            if self.vllm:
                # sampling_params = SamplingParams(max_tokens=4096, temperature=0.7, top_p=0.95)
                sampling_params = SamplingParams(max_tokens=4096, temperature=self.temperature)
                outputs = self.llm.generate(batch_texts, sampling_params=sampling_params, use_tqdm=False)
                batch_responses = [output.outputs[0].text.strip() for output in outputs]
            else:
                # Batch tokenization
                model_inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                ).to(self.llm.device)
                
                # Batch generation
                with torch.no_grad():
                    generated_ids = self.llm.generate(
                        **model_inputs,
                        max_new_tokens=1024,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                
                # Decode batch responses
                batch_responses = []
                for j, generated_id in enumerate(generated_ids):
                    input_length = len(model_inputs.input_ids[j])
                    output_ids = generated_id[input_length:].tolist()
                    response_text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
                    batch_responses.append(response_text)
                    print(f"Batch {i//batch_size + 1}, Sample {j+1}: Generated response")
            
            all_responses.extend(batch_responses)
        
        return all_responses
    
    def evaluate_dataset(self, 
                        dataset_name: str, 
                        num_samples: int = 10, 
                        batch_size: int = 8, 
                        max_workers: int = 4) -> Dict[str, Any]:
        """
        Evaluate a single dataset.
        
        Args:
            dataset_name: Name of the dataset to evaluate
            num_samples: Number of samples to evaluate
            batch_size: Batch size for generation
            max_workers: Number of worker threads for evaluation
            
        Returns:
            Dictionary containing evaluation results and statistics
        """
        if dataset_name not in self.dataset_handlers:
            print(f"❌ Dataset '{dataset_name}' not loaded")
            return {}
        
        handler = self.dataset_handlers[dataset_name]
        print(f"\n--- Evaluating Dataset: {dataset_name} ---")
        
        # Select samples for evaluation
        eval_sample = handler.select_samples(num_samples)
        if eval_sample is None:
            print(f"❌ No samples available for {dataset_name}")
            return {}
        
        # Prepare prompts
        prompts = [handler.prepare_prompt(sample) for sample in eval_sample]
        
        # Generate responses
        all_responses = self.generate_responses(prompts, batch_size)
        
        print(f"Generated all {len(all_responses)} responses. Starting concurrent evaluation...")
        
        # Concurrent evaluation
        def evaluate_single(args):
            idx, prompt, response = args
            print(f"Evaluating sample {idx + 1}/{len(all_responses)}")
            
            sample_data = eval_sample[idx]
            eval_result = handler.evaluate_response(prompt, response, sample_data)
            
            result_dict = {
                "dataset_name": dataset_name,
                "prompt": prompt,
                "response": response,
                "evaluation": eval_result,
            }
            
            # Add any additional sample data
            for key, value in sample_data.items():
                if key not in ["prompt", "dataset_name"]:
                    result_dict[key] = value
            
            return result_dict, eval_result.get("score", 0)
        
        # Use ThreadPoolExecutor for concurrent evaluation
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            evaluation_args = [(i, prompts[i], all_responses[i]) for i in range(len(prompts))]
            results = list(tqdm(executor.map(evaluate_single, evaluation_args), total=len(evaluation_args), desc=f"Evaluating {dataset_name}"))
        
        # Process results
        evaluation_results = []
        for result, score in results:
            evaluation_results.append(result)
        
        print(f"\n--- {dataset_name} Evaluation Complete ---")
        
        # Calculate dataset-specific metrics using the handler
        metrics = handler.calculate_metrics(evaluation_results)
        
        # Print metrics
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name} for {dataset_name}: {metric_value:.4f} on {len(eval_sample)} samples.")
        
        # Save results
        filename = f"evaluation_results-{dataset_name}({self.prefix}).json"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(evaluation_results, f, indent=4)
        print(f"Evaluation results for {dataset_name} saved to {filepath}")
        
        return {
            "dataset_name": dataset_name,
            "metrics": metrics,
            "num_samples": len(eval_sample),
            "results": evaluation_results
        }
    
    def run_evaluation(self, 
                      dataset_names: List[str], 
                      num_samples_to_eval: int = 10, 
                      batch_size: int = 8, 
                      max_workers: int = 4) -> Dict[str, float]:
        """
        Run evaluation on specified datasets.
        
        Args:
            dataset_names: List of dataset names to evaluate
            num_samples_to_eval: Number of samples to evaluate per dataset
            batch_size: Batch size for generation
            max_workers: Number of worker threads for evaluation
            
        Returns:
            Dictionary mapping dataset names to their average scores
        """
        print("\n--- Starting Evaluation ---")
        
        # Load datasets
        self.load_datasets(dataset_names)
        
        if not self.dataset_handlers:
            print("❌ No datasets available for evaluation!")
            return {}
        
        all_results = {}
        
        # Evaluate each dataset
        for dataset_name in self.dataset_handlers.keys():
            try:
                dataset_results = self.evaluate_dataset(
                    dataset_name, 
                    num_samples_to_eval, 
                    batch_size, 
                    max_workers
                )
                all_results[dataset_name] = dataset_results.get("metrics", {})
            except Exception as e:
                print(f"❌ Error evaluating {dataset_name}: {e}")
                all_results[dataset_name] = {}
        
        print(f"\n--- Overall Evaluation Summary ---")
        for dataset_name, metrics in all_results.items():
            if metrics:
                metric_strs = [f"{metric_name}: {metric_value:.4f}" for metric_name, metric_value in metrics.items()]
                print(f"{dataset_name}: {', '.join(metric_strs)}")
            else:
                print(f"{dataset_name}: No metrics available")
        
        return all_results


if __name__ == "__main__":
    # Example usage with dataset selection
    evaluator = ModelEvaluator(
        model_name="/mnt/vm7-home/verl/data/0721-qwen-onlycore/global_step_935",
        output_dir="/mnt/vm7-home/verl/data/0718-qwen",
        prefix="ONLYCORE",
        vllm=True,
        vulnerability_report_path="/mnt/vm7-home/verl/data/0718-qwen/vulnerability_report.json"
    )
    
    # Select which datasets to evaluate
    selected_datasets = ["mmlu", "gsm8k", "strongreject", "harmbench"]
    
    # Run evaluation
    evaluator.run_evaluation(
        dataset_names=selected_datasets,
        num_samples_to_eval=50, 
        batch_size=16, 
        max_workers=4
    )
