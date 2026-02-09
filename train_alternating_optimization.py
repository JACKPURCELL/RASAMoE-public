# Training script with alternating optimization strategy
# 
# STRATEGY:
#   Each round consists of two steps:
#   Step 1: Finetune unsafe experts (freeze routers) using CE loss
#   Step 2: Optimize routers (freeze experts) using TWO-FORWARD consistency loss
#           - Forward safe samples (no_grad) → get target router distributions
#           - Forward unsafe samples (with grad) → match target distributions
#   Repeat for N rounds
#
# KEY DIFFERENCE from train_batch_unsafe_experts.py:
#   - This script uses TWO separate forward passes in router training
#   - Safe router distribution is detached and used as fixed target
#   - Only unsafe router distribution has gradients and is optimized
#   - This ensures safe routing patterns remain unchanged

import os
import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from datasets import Dataset
from typing import List, Dict, Tuple, Optional
from torch.nn import functional as F
from dataclasses import dataclass
import json
from tqdm import tqdm
from torch.utils.data import DataLoader


@dataclass
class DataCollatorWithLabelIds:
    """Custom data collator that handles label_ids (1D) separately from other fields (2D)."""
    tokenizer: AutoTokenizer
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Extract label_ids separately
        label_ids = None
        if 'label_ids' in features[0]:
            label_ids = torch.tensor([f['label_ids'].item() if torch.is_tensor(f['label_ids']) else f['label_ids'] 
                                     for f in features], dtype=torch.long)
            # Remove label_ids from features for default collation
            features = [{k: v for k, v in f.items() if k != 'label_ids'} for f in features]
        
        # Use default collator for other fields
        from transformers import default_data_collator
        batch = default_data_collator(features)
        
        # Add label_ids back
        if label_ids is not None:
            batch['label_ids'] = label_ids
        
        return batch


def load_prepared_batch_data(model_name: str, batch_data_path: str = None, 
                             use_random_experts: bool = False, random_seed: int = 42,
                             unsafe_expert_top_k: int = None) -> Tuple[Dict, List[List[Tuple[int, int]]], List[np.ndarray]]:
    """Load the prepared batch data with PER-BATCH unsafe experts and masks."""
    if batch_data_path is None:
        model_name_clean = model_name.replace('/', '--')
        batch_data_path = f"batch_data_{model_name_clean}.pkl"
    batch_data = pd.read_pickle(batch_data_path)
    
    # Count total examples from batch_data_list
    total_examples = sum(len(batch['batch_outputs']) for batch in batch_data['batch_data_list'])
    print(f"Loaded batch data: {total_examples} examples")
    print(f"Number of batches: {len(batch_data['batch_data_list'])}")
    
    if use_random_experts:
        # Generate random experts instead of using the saved ones
        print(f"\n⚠️  Using RANDOM EXPERTS (ablation study, seed={random_seed})")
        num_layers = batch_data.get('num_layers')
        n_experts = batch_data.get('n_experts')
        if num_layers is None or n_experts is None:
            # Try to infer from batch_data_list
            first_routing = batch_data['batch_data_list'][0]['safe_routings'][0]
            num_layers = first_routing.shape[0]
            n_experts = first_routing.shape[2]
        
        if unsafe_expert_top_k is None:
            # Use the same number as the original batch_unsafe_experts
            unsafe_expert_top_k = len(batch_data['batch_unsafe_experts'][0]) if len(batch_data['batch_unsafe_experts']) > 0 else 50
        
        print(f"  Generating random experts: {unsafe_expert_top_k} experts per batch")
        print(f"  Model dimensions: {num_layers} layers, {n_experts} experts per layer")
        
        batch_unsafe_experts = []
        batch_expert_masks = []
        
        import random
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Generate all possible (layer, expert) pairs
        all_experts = [(layer, expert) for layer in range(num_layers) for expert in range(n_experts)]
        
        for batch_idx in range(len(batch_data['batch_data_list'])):
            # Use batch_idx as part of seed to ensure different random selection per batch
            batch_random_seed = random_seed + batch_idx
            random.seed(batch_random_seed)
            np.random.seed(batch_random_seed)
            
            # Randomly sample top_k experts
            random_experts = random.sample(all_experts, min(unsafe_expert_top_k, len(all_experts)))
            batch_unsafe_experts.append(random_experts)
            
            # Create expert mask
            batch_mask = np.zeros((num_layers, n_experts), dtype=np.float32)
            for layer, expert in random_experts:
                batch_mask[layer, expert] = 1.0
            batch_expert_masks.append(batch_mask)
        
        print(f"  Generated {len(batch_unsafe_experts)} batches with random experts")
    else:
        batch_unsafe_experts = batch_data['batch_unsafe_experts']
        batch_expert_masks = batch_data['batch_expert_masks']
    
    return batch_data, batch_unsafe_experts, batch_expert_masks


def create_batch_structured_dataset(
    sft_data: Dict,
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
    safe_ratio: float = 0.5,
    data_percent: float = 1.0,
) -> List[List[Dict]]:
    """
    Create a dataset that preserves batch structure.
    
    Strategy:
    - Safe prompts: Included in dataset but will be IGNORED during expert training
      (needed for router training contrastive learning)
    - Unsafe prompts: Trained to output refusal response
    
    During expert training, we'll mask out safe samples' loss.
    During router training, both safe and unsafe are needed for contrastive learning.
    """
    batch_data_list = sft_data['batch_data_list']
    all_batches = []
    batch_data_list = batch_data_list[:int(len(batch_data_list) * data_percent)]
    for batch_data in batch_data_list:
        safe_indices = batch_data['safe_indices']
        unsafe_indices = batch_data['unsafe_indices']
        batch_outputs = batch_data['batch_outputs']
        
        batch_examples = []
        
        # Add safe prompts (for router training only, loss will be masked during expert training)
        for idx in safe_indices:
            output = batch_outputs[idx]
            safe_messages = output['messages']
            # We don't add any response here - it won't be trained anyway
            # Just need the messages for router training forward pass
            batch_examples.append({
                "messages": safe_messages,
                "label": "safe",
                "batch_idx": batch_data['batch_idx'],
                "in_batch_idx": len(batch_examples),
                "train_on_this": False  # Mark: don't train on this during expert training
            })
        
        # Add unsafe prompts with refusal response
        for idx in unsafe_indices:
            output = batch_outputs[idx]
            attack_messages = output['messages']
            # Use the GPT-generated refusal response (or fallback to default)
            refusal_response = output.get('refusal_response', "I'm sorry, but I cannot assist with that request as it may be harmful or unsafe.")
            # Append refusal response to the attack messages
            messages = attack_messages + [
                {"role": "assistant", "content": refusal_response}
            ]
            batch_examples.append({
                "messages": messages, 
                "label": "unsafe",
                "batch_idx": batch_data['batch_idx'],
                "in_batch_idx": len(batch_examples),
                "train_on_this": True  # Mark: train on this during expert training
            })
        
        all_batches.append(batch_examples)
    
    total_safe = sum(1 for batch in all_batches for ex in batch if ex['label'] == 'safe')
    total_unsafe = sum(1 for batch in all_batches for ex in batch if ex['label'] == 'unsafe')
    print(f"Created {len(all_batches)} batches with {total_safe} safe (for router training) + {total_unsafe} unsafe (for expert training) examples")
    return all_batches


def create_batch_aware_dataset(
    batches: List[List[Dict]],
    tokenizer: AutoTokenizer,
    max_length: int = 512
) -> Dataset:
    """Convert batch-structured data to HuggingFace Dataset."""
    all_examples = []
    for batch in batches:
        all_examples.extend(batch)
    
    def tokenize_function(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        tokenized["labels"] = tokenized["input_ids"].clone()
        tokenized["batch_idx"] = torch.tensor(examples["batch_idx"], dtype=torch.long)
        
        # Convert label strings to numeric IDs: 0 = safe, 1 = unsafe
        label_ids = torch.tensor([0 if label == "safe" else 1 for label in examples["label"]], dtype=torch.long)
        tokenized["label_ids"] = label_ids
        
        # Add train_on_this flag
        tokenized["train_on_this"] = torch.tensor(examples["train_on_this"], dtype=torch.bool)
        
        return tokenized
    
    dataset_dict = {
        "messages": [ex["messages"] for ex in all_examples],
        "batch_idx": [ex["batch_idx"] for ex in all_examples],
        "label": [ex["label"] for ex in all_examples],
        "train_on_this": [ex["train_on_this"] for ex in all_examples],
    }
    dataset = Dataset.from_dict(dataset_dict)
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset


def freeze_all_except_unsafe_experts(
    model: AutoModelForCausalLM,
    batch_unsafe_experts: List[List[Tuple[int, int]]]
) -> int:
    """Freeze all parameters except unsafe expert MLPs. Returns number of unfrozen experts."""
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
    
    # Get union of all unsafe experts
    all_unsafe_experts = set()
    for batch_experts in batch_unsafe_experts:
        all_unsafe_experts.update(batch_experts)
    
    print(f"Unfreezing unsafe experts: {len(all_unsafe_experts)} total")
    
    # Unfreeze unsafe expert MLPs
    unfrozen_count = 0
    for layer_idx, expert_idx in all_unsafe_experts:
        try:
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                layer = model.model.layers[layer_idx]
                
                if hasattr(layer, "mlp"):
                    mlp = layer.mlp
                    
                    if hasattr(mlp, "experts"):
                        expert = mlp.experts[expert_idx]
                        for param in expert.parameters():
                            param.requires_grad = True
                        unfrozen_count += 1
                    elif hasattr(mlp, "block_sparse_moe") and hasattr(mlp.block_sparse_moe, "experts"):
                        expert = mlp.block_sparse_moe.experts[expert_idx]
                        for param in expert.parameters():
                            param.requires_grad = True
                        unfrozen_count += 1
        except Exception as e:
            print(f"Warning: Could not unfreeze Layer {layer_idx}, Expert {expert_idx}: {e}")
    
    return unfrozen_count


def freeze_all_except_routers(model: AutoModelForCausalLM) -> int:
    """Freeze all parameters except router (gate) parameters. Returns number of unfrozen routers."""
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze router (gate) parameters only
    unfrozen_routers = 0
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer_idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
                
                # OLMoE architecture: mlp.gate (router)
                if hasattr(mlp, "gate"):
                    for param in mlp.gate.parameters():
                        param.requires_grad = True
                    unfrozen_routers += 1
                
                # Mixtral architecture: mlp.block_sparse_moe.gate
                elif hasattr(mlp, "block_sparse_moe") and hasattr(mlp.block_sparse_moe, "gate"):
                    for param in mlp.block_sparse_moe.gate.parameters():
                        param.requires_grad = True
                    unfrozen_routers += 1
    
    return unfrozen_routers


def unfreeze_all_parameters(model: AutoModelForCausalLM) -> int:
    """Unfreeze all model parameters for full finetuning. Returns number of trainable parameters."""
    trainable_params = 0
    for param in model.parameters():
        param.requires_grad = True
        trainable_params += param.numel()
    
    return trainable_params


def register_router_hooks(model: AutoModelForCausalLM, storage: List) -> int:
    """Register forward hooks to capture router logits. Returns number of hooks registered."""
    hook_count = 0
    
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer_idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
                
                # OLMoE architecture
                if hasattr(mlp, "gate"):
                    def make_hook(layer_idx):
                        def hook_fn(module, input, output):
                            logits = output.clone() if torch.is_tensor(output) else output[0].clone()
                            storage.append({
                                'layer_idx': layer_idx,
                                'logits': logits
                            })
                        return hook_fn
                    
                    mlp.gate.register_forward_hook(make_hook(layer_idx))
                    hook_count += 1
                
                # Mixtral architecture
                elif hasattr(mlp, "block_sparse_moe") and hasattr(mlp.block_sparse_moe, "gate"):
                    def make_hook(layer_idx):
                        def hook_fn(module, input, output):
                            logits = output.clone() if torch.is_tensor(output) else output[0].clone()
                            storage.append({
                                'layer_idx': layer_idx,
                                'logits': logits
                            })
                        return hook_fn
                    
                    mlp.block_sparse_moe.gate.register_forward_hook(make_hook(layer_idx))
                    hook_count += 1
    
    return hook_count


def extract_router_probs_from_storage(
    router_logits_storage: List,
    batch_size: int,
    device: torch.device
) -> Dict[int, torch.Tensor]:
    """
    Extract and process router probabilities from storage.
    
    Returns:
        Dict mapping layer_idx -> averaged router probabilities [batch_size, num_experts]
    """
    layer_probs = {}
    
    # Group router logits by layer
    layer_logits = {}
    for item in router_logits_storage:
        layer_idx = item['layer_idx']
        logits = item['logits']
        
        if layer_idx not in layer_logits:
            layer_logits[layer_idx] = []
        layer_logits[layer_idx].append(logits)
    
    # Process each layer
    for layer_idx, logits_list in layer_logits.items():
        all_logits = torch.cat(logits_list, dim=0)  # [batch_size * seq_len, num_experts]
        
        seq_len = all_logits.shape[0] // batch_size
        
        # Reshape and average over sequence: [batch_size, seq_len, num_experts] -> [batch_size, num_experts]
        all_logits = all_logits.reshape(batch_size, seq_len, -1).mean(dim=1)
        
        # Convert to probabilities
        probs = F.softmax(all_logits, dim=-1)  # [batch_size, num_experts]
        
        layer_probs[layer_idx] = probs.to(device)
    
    return layer_probs


def compute_router_consistency_loss_two_forward(
    safe_router_probs: Dict[int, torch.Tensor],
    unsafe_router_probs: Dict[int, torch.Tensor],
    device: torch.device,
    kl_type: str = "forward"
) -> torch.Tensor:
    """
    Compute KL divergence between safe (target) and unsafe router distributions.
    Uses two-forward strategy: safe_probs are detached targets.
    
    Args:
        safe_router_probs: Dict[layer_idx -> probs], detached, as target
        unsafe_router_probs: Dict[layer_idx -> probs], trainable
        device: Device for computation
        kl_type: Type of KL divergence
        
    Returns:
        KL divergence loss
    """
    if len(safe_router_probs) == 0 or len(unsafe_router_probs) == 0:
        return torch.tensor(0.0, device=device)
    
    total_kl_loss = torch.tensor(0.0, device=device)
    num_layers = 0
    
    # Compute KL divergence for each layer
    for layer_idx in safe_router_probs.keys():
        if layer_idx not in unsafe_router_probs:
            continue
        
        safe_probs = safe_router_probs[layer_idx]  # [n_safe, num_experts], detached
        unsafe_probs = unsafe_router_probs[layer_idx]  # [n_unsafe, num_experts], trainable
        
        # Compute average distributions
        safe_avg_probs = safe_probs.mean(dim=0)  # [num_experts]
        unsafe_avg_probs = unsafe_probs.mean(dim=0)  # [num_experts]
        
        # Add epsilon for stability
        eps = 1e-8
        safe_avg_probs = safe_avg_probs + eps
        unsafe_avg_probs = unsafe_avg_probs + eps
        safe_avg_probs = safe_avg_probs / safe_avg_probs.sum()
        unsafe_avg_probs = unsafe_avg_probs / unsafe_avg_probs.sum()
        
        # Compute KL divergence
        # Note: safe_avg_probs is already detached (from torch.no_grad() context)
        # Only unsafe_avg_probs has gradients and will be updated
        if kl_type == "forward":
            # KL(safe || unsafe): push unsafe to match safe (safe is fixed target)
            # This is the RECOMMENDED setting for two-forward strategy
            kl_div = F.kl_div(unsafe_avg_probs.log(), safe_avg_probs, reduction='batchmean')
        elif kl_type == "reverse":
            # KL(unsafe || safe): mathematically still only updates unsafe (safe is detached)
            # But the gradient direction is different - may push unsafe AWAY from safe
            kl_div = F.kl_div(safe_avg_probs.log(), unsafe_avg_probs, reduction='batchmean')
        else:  # symmetric
            # Average of both directions - still only updates unsafe
            kl_forward = F.kl_div(unsafe_avg_probs.log(), safe_avg_probs, reduction='batchmean')
            kl_reverse = F.kl_div(safe_avg_probs.log(), unsafe_avg_probs, reduction='batchmean')
            kl_div = (kl_forward + kl_reverse) / 2.0
        
        total_kl_loss += kl_div.to(device)
        num_layers += 1
    
    if num_layers > 0:
        total_kl_loss = total_kl_loss / num_layers
    
    return total_kl_loss


def train_one_epoch_experts(
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    round_idx: int,
) -> float:
    """
    Train unsafe experts for one epoch (freeze routers).
    Only trains on samples where train_on_this=True (unsafe prompts).
    Safe prompts are skipped during expert training but kept for router training.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Round {round_idx} - Epoch {epoch} - Expert Training (unsafe only)")
    
    for batch in progress_bar:
        # Move batch to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # Extract train_on_this mask
        train_mask = batch.get('train_on_this', None)
        
        if train_mask is None or train_mask.sum() == 0:
            # Skip if no training samples in this batch
            continue
        
        # Filter to only training samples (unsafe prompts)
        inputs = {
            k: v[train_mask] for k, v in batch.items() 
            if k not in ['label_ids', 'train_on_this'] and torch.is_tensor(v)
        }
        
        # Forward pass (only on unsafe prompts)
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({'loss': loss.item(), 'n_unsafe': train_mask.sum().item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def train_one_epoch_router(
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    round_idx: int,
    router_logits_storage: List,
    kl_type: str = "forward"
) -> float:
    """
    Train router for one epoch using consistency loss (freeze experts).
    Uses TWO-FORWARD strategy:
      1. Forward safe samples (no_grad) to get target distributions
      2. Forward unsafe samples (with grad) to compute loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Round {round_idx} - Epoch {epoch} - Router Training")
    
    for batch in progress_bar:
        # Move batch to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # Extract label_ids to separate safe and unsafe samples
        label_ids = batch['label_ids']  # 0 = safe, 1 = unsafe
        
        safe_mask = label_ids == 0
        unsafe_mask = label_ids == 1
        
        # Skip if no safe or unsafe samples
        if safe_mask.sum() == 0 or unsafe_mask.sum() == 0:
            continue
        
        # Prepare inputs (remove label_ids and train_on_this)
        inputs = {k: v for k, v in batch.items() if k not in ['label_ids', 'train_on_this']}
        
        # ============================================================
        # FORWARD 1: Safe samples (no gradient) - get target distribution
        # ============================================================
        safe_inputs = {k: v[safe_mask] for k, v in inputs.items()}
        safe_batch_size = safe_mask.sum().item()
        
        router_logits_storage.clear()
        with torch.no_grad():
            _ = model(**safe_inputs)
        
        # Extract safe router probabilities (detached)
        safe_router_probs = extract_router_probs_from_storage(
            router_logits_storage,
            safe_batch_size,
            device
        )
        
        # ============================================================
        # FORWARD 2: Unsafe samples (with gradient) - trainable
        # ============================================================
        unsafe_inputs = {k: v[unsafe_mask] for k, v in inputs.items()}
        unsafe_batch_size = unsafe_mask.sum().item()
        
        router_logits_storage.clear()
        _ = model(**unsafe_inputs)
        
        # Extract unsafe router probabilities (trainable)
        unsafe_router_probs = extract_router_probs_from_storage(
            router_logits_storage,
            unsafe_batch_size,
            device
        )
        
        # ============================================================
        # Compute router consistency loss
        # ============================================================
        router_loss = compute_router_consistency_loss_two_forward(
            safe_router_probs,   # Target (detached)
            unsafe_router_probs, # Trainable
            device,
            kl_type
        )
        
        if router_loss.item() > 0:
            # Backward pass
            optimizer.zero_grad()
            router_loss.backward()
            optimizer.step()
            
            total_loss += router_loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'router_loss': router_loss.item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

# Example Usage:
#
# 1. Full alternating optimization (expert + router):
# python train_alternating_optimization.py \
#     --model_name allenai/OLMoE-1B-7B-0125-Instruct \
#     --output_dir ./alternating_opt_results \
#     --num_rounds 3 \
#     --epochs_per_round 1 \
#     --expert_lr 5e-5 \
#     --router_lr 1e-4 \
#     --kl_type forward
#
# 2. Expert-only training (skip router training):
# python train_alternating_optimization.py \
#     --model_name allenai/OLMoE-1B-7B-0125-Instruct \
#     --output_dir ./expert_only_results \
#     --num_rounds 3 \
#     --epochs_per_round 1 \
#     --expert_lr 5e-5 \
#     --skip_router_training
#
# 3. Full parameter finetuning (no expert selection):
# python train_alternating_optimization.py \
#     --model_name allenai/OLMoE-1B-7B-0125-Instruct \
#     --output_dir ./full_param_results \
#     --num_rounds 3 \
#     --epochs_per_round 1 \
#     --expert_lr 5e-5 \
#     --full_param_training 

def train_alternating_optimization(
    model_name: str,
    output_dir: str = "./alternating_optimization",
    num_rounds: int = 3,
    expert_epochs_per_round: int = 1,
    router_epochs_per_round: int = 1,
    batch_size: int = 16,
    safe_ratio: float = 0.5,
    expert_lr: float = 5e-5,
    router_lr: float = 1e-4,
    kl_type: str = "forward",
    skip_expert_training: bool = False,
    skip_router_training: bool = False,
    full_param_training: bool = False,
    batch_data_path: str = None,
    data_percent: float = 1.0,
    use_random_experts: bool = False,
    random_seed: int = 42,
    unsafe_expert_top_k: int = None,
):
    """
    Main training function with alternating optimization.
    
    Strategy:
        - If full_param_training=True:
            * Finetune ALL model parameters (no expert selection)
            * Uses CE loss only, skip router training
        
        - If full_param_training=False (default):
            * Round 1, 2, ..., N:
                - Step 1: Finetune unsafe experts (freeze routers) using CE loss
                - Step 2: Optimize routers (freeze experts) using TWO-FORWARD consistency loss
                    - Forward safe samples (no_grad) to get target router distributions
                    - Forward unsafe samples (with grad) to match target distributions
                    - SKIPPED if skip_router_training=True
    
    Args:
        model_name: HuggingFace model name
        output_dir: Directory to save checkpoints
        num_rounds: Number of alternating optimization rounds
        expert_epochs_per_round: Number of epochs per round for expert training
        router_epochs_per_round: Number of epochs per round for router training
        batch_size: Training batch size
        safe_ratio: Ratio of safe prompts in each batch
        expert_lr: Learning rate for expert finetuning
        router_lr: Learning rate for router optimization
        kl_type: Type of KL divergence ("forward" = push unsafe to match safe)
        skip_expert_training: If True, skip expert training (only train routers)
        skip_router_training: If True, skip router training (only train experts)
        full_param_training: If True, finetune all parameters (no expert selection, skip router training)
    """
    print("=" * 80)
    if full_param_training:
        print("FULL PARAMETER FINETUNING")
        print(f"  Mode: All model parameters (no expert selection)")
    elif skip_expert_training and skip_router_training:
        print("WARNING: Both expert and router training are skipped!")
    elif skip_expert_training:
        print("ROUTER-ONLY TRAINING")
    elif skip_router_training:
        print("EXPERT-ONLY TRAINING")
    else:
        print("ALTERNATING OPTIMIZATION TRAINING")
    print(f"  Rounds: {num_rounds}")
    print(f"  Expert epochs per round: {expert_epochs_per_round}")
    print(f"  Router epochs per round: {router_epochs_per_round}")
    print(f"  Expert LR: {expert_lr}, Router LR: {router_lr}")
    if full_param_training:
        print(f"  Expert Training: ENABLED (full param mode)")
        print(f"  Router Training: SKIPPED (full param mode)")
    else:
        print(f"  Expert Training: {'SKIPPED' if skip_expert_training else 'ENABLED'}")
        if skip_router_training:
            print(f"  Router Training: SKIPPED")
        else:
            print(f"  Router Training: ENABLED (KL type: {kl_type})")
    print("=" * 80)
    
    # Load data
    print("\nLoading prepared batch data...")
    batch_data, batch_unsafe_experts, batch_expert_masks = load_prepared_batch_data(
        model_name, batch_data_path, 
        use_random_experts=use_random_experts, 
        random_seed=random_seed,
        unsafe_expert_top_k=unsafe_expert_top_k
    )
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    device = next(model.parameters()).device
    
    # Prepare dataset
    print("\nPreparing dataset...")
    batches = create_batch_structured_dataset(batch_data, tokenizer, batch_size, safe_ratio, data_percent)
    train_dataset = create_batch_aware_dataset(batches, tokenizer)
    
    # Create dataloader
    data_collator = DataCollatorWithLabelIds(tokenizer=tokenizer)
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=True,
    )
    
    # Register router hooks (for router training phase)
    router_logits_storage = []
    if not skip_router_training and not full_param_training:
        num_hooks = register_router_hooks(model, router_logits_storage)
        print(f"Registered {num_hooks} router hooks")
    else:
        print("Router hooks NOT registered (router training skipped)")
    
    # Training loop
    training_logs = []
    
    for round_idx in range(num_rounds):
        print("\n" + "=" * 80)
        print(f"ROUND {round_idx + 1}/{num_rounds}")
        print("=" * 80)
        
        # ====================
        # Step 1: Finetune experts (full params or unsafe experts only)
        # ====================
        if not skip_expert_training:
            if full_param_training:
                print(f"\n[Round {round_idx + 1}] Step 1: Finetuning all parameters...")
                trainable_params = unfreeze_all_parameters(model)
                total_params = sum(p.numel() for p in model.parameters())
                print(f"  Training Mode: Full parameter finetuning")
                print(f"  Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.4f}%)")
            else:
                print(f"\n[Round {round_idx + 1}] Step 1: Finetuning unsafe experts...")
                unfrozen_experts = freeze_all_except_unsafe_experts(model, batch_unsafe_experts)
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())
                print(f"  Unfrozen experts: {unfrozen_experts}")
                print(f"  Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.4f}%)")
            
            # Create optimizer for experts
            expert_optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=expert_lr
            )
            
            # Train experts
            for epoch in range(expert_epochs_per_round):
                expert_loss = train_one_epoch_experts(
                    model, dataloader, expert_optimizer, device, epoch + 1, round_idx + 1
                )
                print(f"  [Round {round_idx + 1}] Expert Training Epoch {epoch + 1}: Loss = {expert_loss:.4f}")
                training_logs.append({
                    'round': round_idx + 1,
                    'step': 'expert',
                    'epoch': epoch + 1,
                    'loss': expert_loss
                })
        else:
            print(f"\n[Round {round_idx + 1}] Step 1: Expert training SKIPPED")
        
        # ====================
        # Step 2: Optimize router consistency (freeze experts)
        # ====================
        if not skip_router_training and not full_param_training:
            print(f"\n[Round {round_idx + 1}] Step 2: Optimizing router consistency...")
            unfrozen_routers = freeze_all_except_routers(model)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  Unfrozen routers: {unfrozen_routers}")
            print(f"  Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.4f}%)")
            
            # Create optimizer for routers
            router_optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=router_lr
            )
            
            # Train routers
            for epoch in range(router_epochs_per_round):
                router_loss = train_one_epoch_router(
                    model, dataloader, router_optimizer, device, epoch + 1, round_idx + 1,
                    router_logits_storage, kl_type
                )
                print(f"  [Round {round_idx + 1}] Router Training Epoch {epoch + 1}: Loss = {router_loss:.4f}")
                training_logs.append({
                    'round': round_idx + 1,
                    'step': 'router',
                    'epoch': epoch + 1,
                    'loss': router_loss
                })
        else:
            print(f"\n[Round {round_idx + 1}] Step 2: Router training SKIPPED")
        
        # Save checkpoint after each round
        round_dir = os.path.join(output_dir, f"round_{round_idx + 1}")
        os.makedirs(round_dir, exist_ok=True)
        model.save_pretrained(round_dir)
        tokenizer.save_pretrained(round_dir)
        print(f"\n  Saved checkpoint to {round_dir}")
    
    # Save final model
    print("\n" + "=" * 80)
    print("Saving final model...")
    print("=" * 80)
    final_dir = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    # Save training logs and metadata
    if full_param_training:
        training_strategy = "full_parameter_finetuning"
    elif skip_expert_training and skip_router_training:
        training_strategy = "no_training"
    elif skip_expert_training:
        training_strategy = "router_only"
    elif skip_router_training:
        training_strategy = "expert_only"
    else:
        training_strategy = "alternating_optimization"
    
    metadata = {
        "model_name": model_name,
        "training_strategy": training_strategy,
        "num_rounds": num_rounds,
        "expert_epochs_per_round": expert_epochs_per_round,
        "router_epochs_per_round": router_epochs_per_round,
        "batch_size": batch_size,
        "safe_ratio": safe_ratio,
        "expert_lr": expert_lr if not skip_expert_training else None,
        "router_lr": router_lr if not skip_router_training and not full_param_training else None,
        "kl_type": kl_type if not skip_router_training and not full_param_training else None,
        "skip_expert_training": skip_expert_training,
        "skip_router_training": skip_router_training,
        "full_param_training": full_param_training,
        "num_batches": len(batches),
        "training_logs": training_logs,
    }
    
    metadata_path = os.path.join(output_dir, "training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    print("\n" + "=" * 80)
    print("Alternating optimization training complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train with alternating optimization")
    parser.add_argument(
        "--model_name",
        type=str,
        default="allenai/OLMoE-1B-7B-0125-Instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./alternating_optimization",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=3,
        help="Number of alternating optimization rounds"
    )
    parser.add_argument(
        "--expert_epochs_per_round",
        type=int,
        default=1,
        help="Number of epochs per round for expert training"
    )
    parser.add_argument(
        "--router_epochs_per_round",
        type=int,
        default=1,
        help="Number of epochs per round for router training"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--safe_ratio",
        type=float,
        default=0.5,
        help="Ratio of safe prompts in each batch"
    )
    parser.add_argument(
        "--expert_lr",
        type=float,
        default=5e-5,
        help="Learning rate for expert finetuning"
    )
    parser.add_argument(
        "--router_lr",
        type=float,
        default=1e-4,
        help="Learning rate for router optimization"
    )
    parser.add_argument(
        "--kl_type",
        type=str,
        default="forward",
        choices=["forward", "reverse", "symmetric"],
        help="Type of KL divergence for router consistency (forward=push unsafe to match safe)"
    )
    parser.add_argument(
        "--skip_expert_training",
        action="store_true",
        help="Skip expert training, only train routers (for ablation study)"
    )
    parser.add_argument(
        "--skip_router_training",
        action="store_true",
        help="Skip router training, only train experts (for ablation study)"
    )
    parser.add_argument(
        "--full_param_training",
        action="store_true",
        help="Enable full parameter finetuning (train all parameters, no expert selection)"
    )
    parser.add_argument(
        "--batch_data_path",
        type=str,
        default=None,
        required=True,
        help="Path to the batch data pickle file"
    )
    
    parser.add_argument(
        "--data_percent",
        type=float,
        default=1.0,
        help="Percentage of data to use for training"
    )
    parser.add_argument(
        "--use_random_experts",
        action="store_true",
        help="Use random experts instead of risk-diff based selection (ablation study)"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for random expert selection"
    )
    parser.add_argument(
        "--unsafe_expert_top_k",
        type=int,
        default=None,
        help="Number of random experts to select per batch (if None, uses the same number as saved in batch_data)"
    )
    args = parser.parse_args()
    
    train_alternating_optimization(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_rounds=args.num_rounds,
        data_percent=args.data_percent,
        expert_epochs_per_round=args.expert_epochs_per_round,
        router_epochs_per_round=args.router_epochs_per_round,
        batch_size=args.batch_size,
        safe_ratio=args.safe_ratio,
        expert_lr=args.expert_lr,
        router_lr=args.router_lr,
        kl_type=args.kl_type,
        skip_expert_training=args.skip_expert_training,
        skip_router_training=args.skip_router_training,
        full_param_training=args.full_param_training,
        batch_data_path=args.batch_data_path,
        use_random_experts=args.use_random_experts,
        random_seed=args.random_seed,
        unsafe_expert_top_k=args.unsafe_expert_top_k,
    )

