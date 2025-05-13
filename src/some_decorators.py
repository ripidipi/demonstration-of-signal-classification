import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from colorama import Fore, Style, init
from dataset import get_dataloaders
from model import CNNClassifier


init(autoreset=True)

def print_header(text):
    print(f"\n{Fore.CYAN}{Style.BRIGHT}==={text}==={Style.RESET_ALL}\n")

def print_success(text):
    print(f"{Fore.GREEN}{text}{Style.RESET_ALL}")

def print_warning(text):
    print(f"{Fore.YELLOW}{text}{Style.RESET_ALL}")

def print_epoch_progress(epoch, total_epochs, train_loss, train_acc, val_loss, val_acc):
    progress = f"Epoch {epoch+1:03d}/{total_epochs}"
    train_stats = f"Train: {Fore.BLUE}Loss {train_loss:.4f}{Style.RESET_ALL} | {Fore.GREEN}Acc {train_acc:.2%}{Style.RESET_ALL}"
    val_stats = f"Val: {Fore.MAGENTA}Loss {val_loss:.4f}{Style.RESET_ALL} | {Fore.CYAN}Acc {val_acc:.2%}{Style.RESET_ALL}"
    print(f"\n{progress} | {train_stats} | {val_stats}")