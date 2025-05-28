import os
import time
import json
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from loader import get_dataloaders
from model import SOTAClassifier
from save import load_checkpoint
from some_decorators import print_header, print_success

GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

os.makedirs("samples", exist_ok=True)

def visualize_signal(iq, true_label, pred_label, idx, out_dir="samples"):
    is_correct = (true_label == pred_label)
    title_color_plot = "green" if is_correct else "red"

    plt.figure(figsize=(10, 4))
    plt.plot(iq[0], label='I (In-phase)')
    plt.plot(iq[1], label='Q (Quadrature)')
    plt.title(
        f"Sample #{idx} | True: {true_label} | Predicted: {pred_label}",
        color=title_color_plot
    )
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()

    fname = os.path.join(out_dir, f"sample_{idx}_vis.png")
    plt.savefig(fname)
    plt.close()


def run_demo(model, le, val_data, n_samples=5):
    print_header("Demo: random samples...")
    for i in range(n_samples):
        idx = random.randint(0, len(val_data) - 1)
        sample = val_data[idx]
        iq1 = sample['iq1'].unsqueeze(0)
        iq2 = sample['iq2'].unsqueeze(0)
        const = sample['const'].unsqueeze(0)
        snr = sample['snr'].unsqueeze(0).unsqueeze(1)
        label_idx = sample['label']

        start = time.time()
        with torch.no_grad():
            out = model(iq1.to(device), iq2.to(device), const.to(device), snr.to(device))
        latency = time.time() - start

        pred_idx = out.argmax(dim=1).item()
        true_label = le.classes_[label_idx]
        pred_label = le.classes_[pred_idx]
        correct = (true_label == pred_label)
        color = GREEN if correct else RED

        print(f"[{i+1}] True: {true_label} | Pred: {pred_label} | time: {latency*1000:.1f} ms")
        visualize_signal(iq1.squeeze(0).cpu().numpy(), true_label, pred_label, i+1)

    print_success("Demo complete.")


def run_metrics(model, le, val_loader):
    print_header("Full evaluation on validation set...")
    all_trues, all_preds = [], []
    latencies = []
    snr_bins = defaultdict(lambda: {'correct':0, 'total':0})

    model.eval()
    for batch in val_loader:
        iq1 = batch['iq1'].to(device)
        iq2 = batch['iq2'].to(device)
        const = batch['const'].to(device)
        snr = batch['snr'].unsqueeze(1).to(device)
        labels = batch['label'].cpu().numpy()

        start = time.time()
        with torch.no_grad():
            out = model(iq1, iq2, const, snr)
        latencies.append((time.time() - start) / iq1.size(0))

        preds = out.argmax(dim=1).cpu().numpy()
        all_trues.extend(labels.tolist())
        all_preds.extend(preds.tolist())

        for lbl, p, s in zip(labels, preds, batch['snr'].numpy()):
            snr_bins[s]['total'] += 1
            if lbl == p:
                snr_bins[s]['correct'] += 1

    acc = accuracy_score(all_trues, all_preds)
    cls_report = classification_report(all_trues, all_preds, target_names=le.classes_, zero_division=0)
    cm = confusion_matrix(all_trues, all_preds)
    avg_latency = np.mean(latencies) * 1000

    per_class_acc = {cls: cm[i,i] / cm[i].sum() if cm[i].sum()>0 else 0.0
                     for i, cls in enumerate(le.classes_)}

    snr_acc = {str(snr): bins['correct']/bins['total']
               for snr, bins in snr_bins.items() if bins['total']>0}

    report = {
        'overall_accuracy': acc,
        'per_class_accuracy': per_class_acc,
        'classification_report': cls_report,
        'confusion_matrix': cm.tolist(),
        'average_latency_ms': avg_latency,
        'snr_accuracy': snr_acc
    }

    print(f"Overall Accuracy: {acc:.4f}")
    print("Per-class Accuracy:")
    for cls, a in per_class_acc.items():
        print(f"  {cls}: {a:.4f}")
    print("\nClassification Report:\n", cls_report)
    print("Average Latency (ms/sample):", f"{avg_latency:.2f}")
    print("Accuracy by SNR:")
    for s, a in snr_acc.items():
        print(f"  SNR={s}: {a:.4f}")

    out_path = os.path.join("samples", "metrics_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print_success(f"Metrics saved to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    _, val_loader, num_classes = get_dataloaders("data/raw/RML2016.10a_dict.pkl")
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    model = SOTAClassifier(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    model, optimizer, _, le = load_checkpoint(model, optimizer, args.checkpoint, device)

    val_data = list(val_loader.dataset)

    if args.demo:
        run_demo(model, le, val_data, n_samples=args.samples)
    if args.metrics:
        run_metrics(model, le, val_loader)
    if not args.demo and not args.metrics:
        parser.print_help()