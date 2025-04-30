"""
Explore what metadata is saved in n2v checkpoints
"""

import torch
import json

# Load the checkpoint
checkpoint_path = "../data/models/n2v/checkpoints/last-v1.ckpt"
checkpoint = torch.load(checkpoint_path)

# Print out the keys in the checkpoint
print("Checkpoint Keys:", checkpoint.keys())

# Inspect model state_dict
if "state_dict" in checkpoint:
    print("\nModel State Dict:")
    for key, tensor in checkpoint["state_dict"].items():
        print(f"{key}: {tensor.shape}")

# Inspect optimizer state_dict if available
if "optimizer_state_dict" in checkpoint:
    print("\nOptimizer State Dict:")
    optimizer_state = checkpoint["optimizer_state_dict"]
    for key, value in optimizer_state.items():
        # Some entries in the optimizer state might be tensors, numbers, or dicts.
        print(f"{key}: {value}")

# Check additional metadata such as epoch
if "epoch" in checkpoint:
    print("\nCheckpoint saved at epoch:", checkpoint["epoch"])

# Check which hyperparameters were saved
if "hyper_parameters" in checkpoint:
    hyperparams = json.dumps(checkpoint["hyper_parameters"], indent=4)
    print(hyperparams)
