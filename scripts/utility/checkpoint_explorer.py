"""
Check timestamps of the saved model checkpoints.
"""

from pathlib import Path
import torch
import os
from datetime import datetime

# Load the checkpoint
checkpoint_dir = Path("../data/models/n2v/checkpoints")
checkpoint_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getctime)
for checkpoint_path in checkpoint_list:
    checkpoint = torch.load(checkpoint_path)
    creation_time = datetime.fromtimestamp(os.path.getctime(checkpoint_path))
    modification_time = datetime.fromtimestamp(os.path.getmtime(checkpoint_path))
    epoch = checkpoint["epoch"]
    print(
        f"Checkpoint {checkpoint_path.name}: epoch {epoch} saved at {creation_time}, modified at {modification_time}"
    )
