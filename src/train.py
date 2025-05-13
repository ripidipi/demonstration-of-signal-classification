import os
import argparse
import matplotlib.pyplot as plt
from tqdm import trange
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import CNNClassifier
from save import load_checkpoint, save_checkpoint
from some_decorators import print_epoch_progress, print_header, print_success, print_warning

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        correct += (output.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y)

            total_loss += loss.item() * X.size(0)
            correct += (output.argmax(dim=1) == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total

def plot_history(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train")
    plt.plot(val_accs, label="Val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_progress.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, help="Path to checkpoint")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() 
        else "cuda" if torch.cuda.is_available() 
        else "cpu"
    )
    print_header(f"Using device: {device}")

    data_path = "data/raw/RML2016.10a_dict.pkl"
    train_loader, val_loader, le = get_dataloaders(data_path)

    if args.resume:
        model = CNNClassifier(num_classes=len(le.classes_)).to(device)
        optimizer = optim.Adam(model.parameters())  
        model, optimizer, start_epoch, _ = load_checkpoint(
            model, optimizer, args.resume, device
        )
        print_success(f"Resuming from epoch {start_epoch}")
    else:
        model = CNNClassifier(num_classes=len(le.classes_)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-5)
        start_epoch = 0
        print_success("Starting new training session")

    criterion = nn.CrossEntropyLoss()

    sample, _ = next(iter(train_loader))
    print(f"Input shape: {sample.shape}")
    print(f"Output shape: {model(sample.to(device)).shape}")

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_acc = 0.0
    total_epochs = 40

    print_header("=== Starting Training ===")

    try:
        for epoch in trange(start_epoch, total_epochs, desc="Training Progress"):
            tr_loss, tr_acc = train(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            train_losses.append(tr_loss)
            val_losses.append(val_loss)
            train_accs.append(tr_acc)
            val_accs.append(val_acc)

            print_epoch_progress(epoch, total_epochs, tr_loss, tr_acc, val_loss, val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(model, optimizer, epoch + 1, val_loss, val_acc, le.classes_, args.save_dir, is_best=True)
                print_success(f"New best accuracy: {val_acc:.2%} - model saved!")

            if (epoch + 1) % 10 == 0:
                save_checkpoint(model, optimizer, epoch + 1, val_loss, val_acc, le.classes_, args.save_dir, is_best=False)

        plot_history(train_losses, val_losses, train_accs, val_accs)
        print_header("Training Complete!")

    except KeyboardInterrupt:
        print_warning("Training interrupted! Saving current state...")
        save_checkpoint(model, optimizer, epoch + 1, val_loss, val_acc, le.classes_, args.save_dir, is_best=False)
        print_success("Checkpoint saved.")

if __name__ == "__main__":
    main()

# Usage 
# python src/train.py 
# python src/train.py --resume checkpoints/best_model.pth           
