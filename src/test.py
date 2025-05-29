import os
import time
import json
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from loader import get_dataloaders
from model import SOTAClassifier
from save import load_checkpoint
import seaborn as sns 
from some_decorators import print_header, print_success


classes = [
    'OOK','4ASK','8ASK','BPSK','QPSK','8PSK','16PSK','32PSK',
    '16APSK','32APSK','64APSK','128APSK','16QAM','32QAM',
    '64QAM','128QAM','256QAM','AM-SSB-WC','AM-SSB-SC',
    'AM-DSB-WC','AM-DSB-SC','FM','GMSK','OQPSK'
]
le = LabelEncoder()
le.classes_ = np.array(classes)

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("samples", exist_ok=True)



def plot_confusion_matrix(cm, class_names, out_path="samples/confusion_matrix.png"):
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def visualize_signal(iq, true_label, pred_label, idx, out_dir="samples"):
    color = 'green' if true_label == pred_label else 'red'
    plt.figure(figsize=(10,4))
    plt.plot(iq[0])
    plt.plot(iq[1])
    plt.title(f"#{idx} True:{true_label} Pred:{pred_label}", color=color)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"sample_{idx}.png"))
    plt.close()

def run_demo(model, le, val_data, n_samples=5):
    print_header("Demo")
    model.eval()
    for i in tqdm(range(n_samples)):
        idx = random.randint(0, len(val_data)-1)
        sample = val_data[idx]
        iq1, iq2, const, snr, label_idx = [sample[j].unsqueeze(0) for j in range(5)]
        with torch.no_grad():
            out = model(iq1.to(device), iq2.to(device), const.to(device), snr.to(device))
        pred_idx = out.argmax(dim=1).item()
        true_label = le.classes_[label_idx]
        pred_label = le.classes_[pred_idx]
        print(f"{i+1} True:{true_label} Pred:{pred_label}")
        visualize_signal(iq1.squeeze(0).cpu().numpy(), true_label, pred_label, i+1)
    print_success("Demo done")

def run_metrics(model, le, val_loader):
    print_header("Metrics")
    all_t, all_p, lat, snr_bins = [], [], [], defaultdict(lambda:{'c':0,'t':0})
    model.eval()
    for iq1, iq2, const, snr, labels in tqdm(val_loader):
        iq1 = iq1.to(device)
        iq2 = iq2.to(device)
        const = const.to(device)
        snr = snr.view(-1,1).to(device)
        labels_np = labels.cpu().numpy()
        start = time.time()
        with torch.no_grad():
            out = model(iq1, iq2, const, snr)
        lat.append((time.time()-start)/iq1.size(0))
        preds = out.argmax(dim=1).cpu().numpy()
        all_t.extend(labels_np.tolist())
        all_p.extend(preds.tolist())
        for lbl, p, s in zip(labels_np, preds, snr.cpu().numpy().flatten()):
            snr_bins[s]['t'] += 1
            if lbl == p: snr_bins[s]['c'] += 1
    acc = accuracy_score(all_t, all_p)
    report = classification_report(all_t, all_p, target_names=le.classes_, zero_division=0)
    cm = confusion_matrix(all_t, all_p)
    avg_lat = np.mean(lat) * 1000
    per_cls = {cls: cm[i,i]/cm[i].sum() if cm[i].sum()>0 else 0.0 for i,cls in enumerate(le.classes_)}
    snr_acc = {str(s): b['c']/b['t'] for s,b in snr_bins.items() if b['t']>0}
    print(f"Acc:{acc:.4f} Lat(ms):{avg_lat:.2f}")
    print(report)
    out = {'acc': acc, 'per_cls': per_cls, 'snr': snr_acc, 'cm': cm.tolist(), 'report': report}
    plot_confusion_matrix(cm, le.classes_, out_path="samples/confusion_matrix.png")
    with open(os.path.join("samples","metrics.json"), 'w') as f:
        json.dump(out, f)
    print_success("Metrics saved")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    _, val_loader, _ = get_dataloaders("data/raw/radioml2018/versions/2/GOLD_XYZ_OSC.0001_1024.hdf5", batch_size=512)
    model = SOTAClassifier(num_classes=len(classes)).to(device)
    model, le_ckpt = load_checkpoint(model, args.checkpoint, device)
    if le_ckpt is not None:
        le = le_ckpt
    if args.demo:
        run_demo(model, le, list(val_loader.dataset), n_samples=args.samples)
    if args.metrics:
        run_metrics(model, le, val_loader)
    if not args.demo and not args.metrics:
        parser.print_help()
