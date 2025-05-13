import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from datetime import datetime
from sklearn.preprocessing import LabelEncoder  
from dataset import get_dataloaders
import matplotlib.pyplot as plt
from model import CNNClassifier

def save_checkpoint(model, optimizer, epoch, loss, acc, le_classes, save_dir="checkpoints", is_best=False):
    os.makedirs(save_dir, exist_ok=True)
    
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': loss,
        'accuracy': acc,
        'classes': le_classes.tolist(),
        'format_version': 2,
        'timestamp': datetime.now().isoformat()
    }
    
    filename = f"checkpoint_epoch_{epoch}.pth" if not is_best else "best_model.pth"
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'format_version' not in checkpoint:  
        le_classes = checkpoint.get('le_data', {}).get('classes', [])
    else:
        le_classes = checkpoint['classes']
    
    le = LabelEncoder()
    if len(le_classes) > 0:
        le.classes_ = np.array(le_classes)
    
    model.load_state_dict(checkpoint['model_state'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (loss: {checkpoint['loss']:.4f}, acc: {checkpoint['accuracy']:.4f})")
    return model, optimizer, checkpoint['epoch'], le
