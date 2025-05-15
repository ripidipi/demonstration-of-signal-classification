import os
import torch
import matplotlib.pyplot as plt
import random
from loader import get_dataloaders
from model import CNNClassifier
from save import load_checkpoint
from some_decorators import print_header, print_success

GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

os.makedirs("samples", exist_ok=True)

def visualize_signal(signal, true_label, pred_label, idx, out_dir="samples"):
    """
    Save I/Q signal plot with colored title indicating correctness.
    """
    is_correct = (true_label == pred_label)
    title_color_console = GREEN if is_correct else RED
    title_color_plot = "green" if is_correct else "red"

    plt.figure(figsize=(10, 4))
    plt.plot(signal[0], label='I (In-phase)')
    plt.plot(signal[1], label='Q (Quadrature)')
    plt.title(
        f"Sample #{idx} | True: {true_label} | Predicted: {pred_label}",
        color=title_color_plot
    )
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()

    filename = os.path.join(out_dir, f"sample_{idx}_visualization.png")
    plt.savefig(filename)
    plt.close()


def run_demo(checkpoint_path, n_samples=5):
    print_header("Demo: Loading data and model...")
    data_path = "data/raw/RML2016.10a_dict.pkl"
    _, val_loader, le = get_dataloaders(data_path)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() 
        else "cuda" if torch.cuda.is_available() 
        else "cpu"
    )

    model = CNNClassifier(num_classes=len(le.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    model, optimizer, _, _ = load_checkpoint(model, optimizer, checkpoint_path, device)
    model.eval()
    print_success("Model loaded successfully!")

    val_data = list(val_loader.dataset)
    print_header("Running demo predictions...")

    for i in range(n_samples):
        idx = random.randint(0, len(val_data) - 1)
        signal, label = val_data[idx]
        signal_tensor = signal.unsqueeze(0).to(device)
        output = model(signal_tensor)
        pred_idx = output.argmax(dim=1).item()
        true_label = le.classes_[label]
        pred_label = le.classes_[pred_idx]

        is_correct = (true_label == pred_label)
        console_color = GREEN if is_correct else RED
        print(f"{console_color}[{i+1}] True: {true_label} | Predicted: {pred_label}{RESET}")

        visualize_signal(signal.cpu().numpy(), true_label, pred_label, i + 1)

    print_success("Demo complete! Signal visualizations saved.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to demo")
    args = parser.parse_args()

    run_demo(args.checkpoint, args.samples)

# Usage:
# python src/test.py --checkpoint checkpoints/best_model.pth --samples 5