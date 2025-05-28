import matplotlib.pyplot as plt
from colorama import Fore,Style

def print_header(t):print(f"\n{Fore.CYAN}{Style.BRIGHT}=== {t} ==={Style.RESET_ALL}")
def print_success(t):print(f"{Fore.GREEN}{t}{Style.RESET_ALL}")
def plot_history(h):plt.figure();plt.plot(h['tr_acc'],label='tr');plt.plot(h['val_acc'],label='val');plt.legend();plt.show()