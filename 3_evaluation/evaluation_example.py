#!/usr/bin/env python3
"""
Example usage of all attacks dataset evaluation.
"""

import os
import json
import argparse
from evaluation_refactored import ModelEvaluator

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

def main(args):
    """Main function demonstrating all attacks dataset evaluation."""
    
    model_name = args.model_name
    prefix = args.prefix
    output_dir = args.output_dir
    flipattack_path = args.flipattack_path
    deepinception_path = args.deepinception_path
    johnny_path = args.johnny_path
    xteaming_path = args.xteaming_path
    selected_datasets = args.datasets
    tensor_parallel_size = args.tensor_parallel_size
    print("=== Starting all attacks Evaluation ===")
    print(f"Model: {model_name}")
    print(f"Output directory: {output_dir}")
    # print(f"Dataset path: {olmoe_fwo_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize evaluator with all attacks dataset path
    evaluator = ModelEvaluator(
        model_name=model_name,
        output_dir=output_dir,
        prefix=prefix,
        vllm=True,
        flipattack_path=flipattack_path,
        deepinception_path=deepinception_path,
        johnny_path=johnny_path,
        xteaming_path=xteaming_path,
        tensor_parallel_size=tensor_parallel_size
    )
    
    # Run evaluation
    scores = evaluator.run_evaluation(
        dataset_names=selected_datasets,
        num_samples_to_eval=100,  # Evaluate 50 samples as in original eval1124.py
        batch_size=8,
        max_workers=5  # Use 3 workers for concurrent judge evaluation
    )
    
    # Save aggregated scores
    scores_file = os.path.join(output_dir, f"full_scores_{prefix}.json")
    if os.path.exists(scores_file):
        with open(scores_file, "r") as f:
            existing_scores = json.load(f)
        existing_scores.update(scores)
        scores = existing_scores
    
    with open(scores_file, "w") as f:
        json.dump(scores, f, indent=4)
    
    print(f"\n=== Evaluation Complete ===")
    print(f"Full evaluation scores: {scores}")
    print(f"Results saved to: {scores_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="all attacks dataset evaluation")
    parser.add_argument(
        "--model_name",
        type=str,
        default="allenai/OLMoE-1B-7B-0924",
        help="Model name or path"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="OLMoE-0924",
        help="Prefix for output files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--flipattack_path",
        type=str,
        default="~/RASAMoE/FLIPATTACK.json",
        help="Path to FLIPATTACK dataset"
    )
    parser.add_argument(
        "--deepinception_path",
        type=str,
        default="~/RASAMoE/a_exp01/exp_deepinception_0104/data_Qwen_Qwen3_30B_A3B_walledai_AdvBench_test.json",
        help="Path to DeepInception dataset"
    )
    parser.add_argument(
        "--johnny_path",
        type=str,
        default="~/RASAMoE/a_exp01/exp_johnny_0106/qwen-test.json",
        help="Path to Johnny dataset"
    )
    parser.add_argument(
        "--xteaming_path",
        type=str,
        default="~/RASAMoE/0118_exp/xteaming/qwen_test.json",
        help="Path to Xteaming dataset"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["strongreject", "xstest", "olmoe_fwo", "deepinception", "truthfulqa", "johnny", "xteaming", "mmlu", "gsm8k"],
        # default=["strongreject", "xstest", "olmoe_fwo", "deepinception", "truthfulqa"],
        help="List of datasets to evaluate (e.g., strongreject xstest olmoe_fwo)"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of tensor parallel size"
    )
    
    args = parser.parse_args()
    main(args)

