import os
import torch
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

def save_checkpoint(model, optimizer, epoch, loss, acc, le_classes, save_dir="checkpoints", is_best=False):
    os.makedirs(save_dir, exist_ok=True)

    if hasattr(le_classes, 'tolist'):
        classes = le_classes.tolist()
    else:
        classes = list(le_classes)

    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': loss,
        'accuracy': acc,
        'classes': classes,
        'format_version': 2,
        'timestamp': datetime.now().isoformat()
    }

    filename = f"checkpoint_epoch_{epoch}.pth" if not is_best else "best_model.pth"
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)


    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        le = checkpoint.get('le', checkpoint.get('label_encoder', None))
        if le is not None:
            print("Model and label encoder loaded from checkpoint")
        else:
            print("Model loaded from checkpoint; no label encoder found in checkpoint")
    else:
        model.load_state_dict(checkpoint)
        le = None
        print("Model loaded from simple state_dict checkpoint")

    model.eval()
    return model, le

