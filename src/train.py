import os
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from loader import get_dataloaders
from model import SOTAClassifier
from losses import HybridLoss
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='plots/confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def mixup_data(x1, x2, const, snr, y, alpha=0.4):
    if alpha <= 0:
        return x1, x2, const, snr, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x1.size(0), device=x1.device)
    return (lam*x1 + (1-lam)*x1[idx],
            lam*x2 + (1-lam)*x2[idx],
            lam*const + (1-lam)*const[idx],
            lam*snr + (1-lam)*snr[idx],
            y, y[idx], lam)

def train_one_epoch(model, loader, opt, crit, scaler, device, scheduler, mixup_alpha, clip_grad=1.0):
    model.train()
    total_loss = 0.0; total_corr = 0; total_samples = 0

    for x1, x2, const, snr, y in tqdm(loader, desc='Train'):
        x1, x2, const, snr, y = [t.to(device) for t in (x1, x2, const, snr, y)]
        snr = snr.view(-1,1)

        x1, x2, const, snr, y1, y2, lam = mixup_data(x1, x2, const, snr, y, alpha=mixup_alpha)

        opt.zero_grad()
        with autocast(enabled=(scaler is not None)):
            out = model(x1, x2, const, snr)
            loss1 = crit(out, y1)
            loss2 = crit(out, y2)
            loss = lam*loss1 + (1-lam)*loss2
            loss = loss.mean()

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()

        if scheduler:
            scheduler.step()

        preds = out.argmax(dim=1)
        total_corr += (lam*(preds==y1).sum().item() + (1-lam)*(preds==y2).sum().item())
        total_samples += y.size(0)
        total_loss += loss.item() * y.size(0)

    return total_loss/total_samples, 100.*total_corr/total_samples

def validate(model, loader, crit, device):
    model.eval()
    total_loss = 0.0; total_corr = 0; total_samples = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x1, x2, const, snr, y in tqdm(loader, desc='Valid'):
            x1, x2, const, snr, y = [t.to(device) for t in (x1, x2, const, snr, y)]
            snr = snr.view(-1,1)

            out = model(x1, x2, const, snr)
            loss = crit(out, y).mean()
            preds = out.argmax(dim=1)

            total_loss += loss.item() * y.size(0)
            total_corr += (preds==y).sum().item()
            total_samples += y.size(0)

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return total_loss/total_samples, 100.*total_corr/total_samples, all_labels, all_preds

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch',  type=int, default=512)
    parser.add_argument('--lr',     type=float, default=1e-3)
    parser.add_argument('--wd',     type=float, default=1e-3)
    parser.add_argument('--mixup',  type=float, default=0.4)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, num_classes = get_dataloaders(
        'data/raw/radioml2018/versions/2/GOLD_XYZ_OSC.0001_1024.hdf5',
        batch_size=args.batch, num_workers=10, val_fraction=0.2
    )

    model = SOTAClassifier(num_classes).to(device)
    opt   = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit  = HybridLoss(alpha=0.25, gamma=2.0, smoothing=0.1)
    scaler= None if args.no_amp else GradScaler()

    total_steps = args.epochs * len(train_loader)
    sched = OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps,
                       pct_start=0.1, anneal_strategy='cos')

    start_epoch = 1
    best_acc = 0.0
    patience, wait = 10, 0

    if args.resume and os.path.exists('best.pth'):
        ckpt = torch.load('best.pth', map_location=device)
        model.load_state_dict(ckpt)
        print("Loaded checkpoint best.pth")

    for epoch in range(start_epoch, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, crit, scaler, device, sched, args.mixup)
        val_loss, val_acc, y_true, y_pred = validate(model, val_loader, crit, device)

        print(f"[{epoch:03d}] train_loss={tr_loss:.4f} train_acc={tr_acc:.2f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc; wait = 0
            torch.save(model.state_dict(), 'best.pth')
            print(f"→ New best: {best_acc:.2f}%, saving best.pth")
            class_names = train_loader.dataset.y.tolist()
            class_names = [f"C{i}" for i in range(num_classes)]
            plot_confusion_matrix(y_true, y_pred, class_names, save_path='plots/confusion_best.png')
        else:
            wait += 1
            if wait >= patience:
                print(f"No improvement in {patience} epochs, stopping.")
                break

    print("Training complete. Best val_acc: {:.2f}%".format(best_acc))

if __name__ == '__main__':
    main()