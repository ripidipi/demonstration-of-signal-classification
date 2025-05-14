import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from colorama import Fore, Style, init

from dataset import get_dataloaders
from model import CNNClassifier
from save import load_checkpoint, save_checkpoint
from some_decorators import print_header, print_success, print_warning

init(autoreset=True)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, target):
        logp = -self.ce(logits, target)
        p = torch.exp(logp)
        loss = -((1 - p) ** self.gamma) * logp
        return loss.mean()

def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    tot_loss, tot_corr, tot = 0, 0, 0
    pbar = tqdm(loader, desc=f"{Fore.BLUE}Train{Style.RESET_ALL}", leave=False)
    for X, y in pbar:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        preds = logits.argmax(1)
        acc = (preds == y).float().mean().item()
        tot_loss += loss.item()*y.size(0)
        tot_corr += (preds == y).sum().item()
        tot += y.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2%}")
    return tot_loss/tot, tot_corr/tot

def valid_epoch(model, loader, criterion, device):
    model.eval()
    tot_loss, tot_corr, tot = 0, 0, 0
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"{Fore.GREEN}Valid{Style.RESET_ALL}", leave=False)
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            preds = logits.argmax(1)

            acc = (preds == y).float().mean().item()
            tot_loss += loss.item()*y.size(0)
            tot_corr += (preds == y).sum().item()
            tot += y.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2%}")
    return tot_loss/tot, tot_corr/tot

def plot_history(hist, save_path="training_progress.png"):
    plt.figure(figsize=(12,5))
    epochs = range(1, len(hist['tr_loss'])+1)
    plt.subplot(1,2,1)
    plt.plot(epochs, hist['tr_loss'], label='Train')
    plt.plot(epochs, hist['val_loss'], label='Val')
    plt.title('Loss'); plt.xlabel('Epoch'); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, hist['tr_acc'], label='Train')
    plt.plot(epochs, hist['val_acc'], label='Val')
    plt.title('Accuracy'); plt.xlabel('Epoch'); plt.legend()
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--resume', type=str)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    args = p.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available()
                          else 'cpu')
    print_header(f"Using device: {device}")

    tr_loader, val_loader, le = get_dataloaders(
        'data/raw/RML2016.10a_dict.pkl',
        batch_size=256, num_workers=8
    )

    model = CNNClassifier(num_classes=len(le.classes_)).to(device)
    start_epoch = 0
    if args.resume:
        opt = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        model, opt, start_epoch, _ = load_checkpoint(
            model, opt, args.resume, device
        )
        print_success(f"Resumed from epoch {start_epoch}")
    else:
        opt = AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        print_success("Starting new training")

    total_steps = 100 * len(tr_loader)
    scheduler = OneCycleLR(
        opt, max_lr=1e-2, total_steps=total_steps,
        pct_start=0.3, anneal_strategy='cos',
        div_factor=25, final_div_factor=1e4
    )
    criterion = FocalLoss(gamma=2.0)

    history = {'tr_loss':[], 'val_loss':[], 'tr_acc':[], 'val_acc':[]}
    best_acc = 0.0

    for epoch in range(start_epoch, 100):
        print_header(f"Epoch {epoch+1}/100")
        tr_loss, tr_acc = train_epoch(model, tr_loader, opt, scheduler, criterion, device)
        val_loss, val_acc = valid_epoch(model, val_loader, criterion, device)

        history['tr_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['tr_acc'].append(tr_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}: Tr Acc={tr_acc:.2%} | Val Acc={val_acc:.2%}")
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                model, opt, epoch+1, val_loss, val_acc,
                le.classes_, args.save_dir, is_best=True
            )
            print_success(f"Best acc {best_acc:.2%} saved")
        if (epoch+1)%10==0:
            save_checkpoint(
                model, opt, epoch+1, val_loss, val_acc,
                le.classes_, args.save_dir, is_best=False
            )

    plot_history(history)
    print_header("Training finished")

if __name__=='__main__':
    main()
