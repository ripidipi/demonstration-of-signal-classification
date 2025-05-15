import argparse
from tqdm import trange, tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import GradScaler, autocast

from loader import get_dataloaders
from model import CNNClassifier
from save import save_checkpoint


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        logp = -self.ce(logits, targets)
        p = torch.exp(logp)
        return -((1 - p) ** self.gamma * logp).mean()


def train_epoch(model, loader, optimizer, scheduler, criterion, scaler, device, pin_mem, use_amp):
    model.train()
    total_loss = total_correct = total_samples = 0

    # Use per-sample loss for mixup
    ce_none = nn.CrossEntropyLoss(reduction='none')

    for X, y in tqdm(loader, desc='Train ', leave=False):
        X = X.to(device, non_blocking=pin_mem)
        y1, y2, lam = y
        y1 = y1.to(device, non_blocking=pin_mem)
        y2 = y2.to(device, non_blocking=pin_mem)
        lam = lam.to(device, non_blocking=pin_mem)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(X)
            # per-sample cross-entropy
            loss1 = ce_none(logits, y1)
            loss2 = ce_none(logits, y2)
            # weighted combination per sample, then average
            mixed_loss = lam * loss1 + (1 - lam) * loss2
            loss = mixed_loss.mean()

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()

        preds = logits.argmax(dim=1)
        total_correct += (preds == y1).sum().item()
        total_loss += loss.item() * X.size(0)
        total_samples += X.size(0)

    return total_loss / total_samples, total_correct / total_samples


def valid_epoch(model, loader, criterion, device, pin_mem, use_amp):
    model.eval()
    total_loss = total_correct = total_samples = 0
    with torch.no_grad():
        for X, y in tqdm(loader, desc='Valid ', leave=False):
            X = X.to(device, non_blocking=pin_mem)
            y = y.to(device, non_blocking=pin_mem)
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(X)
                loss = criterion(logits, y)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_loss += loss.item() * X.size(0)
            total_samples += X.size(0)
    return total_loss / total_samples, total_correct / total_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, help='Path to checkpoint')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    pin_mem = device.type == 'cuda'
    use_amp = pin_mem and args.amp

    tr_dl, val_dl, label_encoder = get_dataloaders(
        'data/raw/RML2016.10a_dict.pkl',
        batch_size=args.batch,
        pin_memory=pin_mem,
        num_workers=4
    )

    model = CNNClassifier(num_classes=len(label_encoder.classes_)).to(device)
    opt = AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    total_steps = args.epochs * len(tr_dl)
    sched = OneCycleLR(opt, max_lr=1e-2, total_steps=total_steps,
                       pct_start=0.3, div_factor=25, final_div_factor=1e4)
    crit = FocalLoss(gamma=2.0)
    scaler = GradScaler(enabled=use_amp)

    best_acc = 0.0
    for epoch in trange(args.epochs, desc='Epochs'):
        tr_loss, tr_acc = train_epoch(model, tr_dl, opt, sched, crit,
                                      scaler, device, pin_mem, use_amp)
        val_loss, val_acc = valid_epoch(model, val_dl, crit,
                                        device, pin_mem, use_amp)
        print(f"Epoch {epoch+1}/{args.epochs} — Train: loss={tr_loss:.4f}, acc={tr_acc:.2%} | Val: loss={val_loss:.4f}, acc={val_acc:.2%}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, opt, epoch+1,
                            val_loss, val_acc,
                            label_encoder.classes_,
                            is_best=True)
            print(f"New best accuracy: {val_acc:.2%} - model saved!")

if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    main()