import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load Bubble Sort data from the renamed output file
df = pd.read_csv("la2.txt")

N = df["N"].values
bubble_seq = df["SEQ"].values
bubble_par = df["PAR"].values
bubble_speed = df["SPEEDUP"].values
bubble_cost = df["COST"].values
bubble_eff = df["EFFICIENCY"].values
CORES = df["CORES"].values[0]

# ================= BUBBLE TIME & CROSSOVER DETECTION =================
plt.figure(figsize=(8, 5))
plt.plot(N, bubble_seq, marker='o', linewidth=2, color='#1f77b4', label="Bubble Sequential")
plt.plot(N, bubble_par, marker='o', linewidth=2, color='#ff7f0e', label="Bubble Parallel")
plt.xlabel("Input Size N")
plt.ylabel("Time (seconds)")
plt.title("Bubble Sort: Sequential vs Parallel Performance\n(Crossover Detection)", fontsize=12, fontweight='bold')
plt.xticks(N, rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ================= BUBBLE SPEEDUP =================
plt.figure(figsize=(8, 5))
plt.plot(N, bubble_speed, marker='s', linewidth=2, color='#2ca02c', label="Bubble Speedup")
plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label="No Speedup (1.0)")
plt.axhline(y=CORES, color='g', linestyle='-.', alpha=0.5, label=f"Ideal Speedup ({CORES})")
plt.xlabel("Input Size N")
plt.ylabel("Speedup")
plt.title("Bubble Sort: Speedup Analysis", fontsize=12, fontweight='bold')
plt.xticks(N, rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ================= BUBBLE EFFICIENCY =================
plt.figure(figsize=(8, 5))
plt.plot(N, bubble_eff, marker='d', linewidth=2, color='#9467bd', label="Bubble Efficiency")
plt.axhline(y=1.0, color='g', linestyle='--', alpha=0.7, label="Ideal Efficiency (1.0)")
plt.xlabel("Input Size N")
plt.ylabel("Efficiency")
plt.title("Bubble Sort: Efficiency Analysis", fontsize=12, fontweight='bold')
plt.xticks(N, rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ================= COST GRAPH =================
plt.figure(figsize=(8, 5))
plt.plot(N, bubble_cost, marker='p', linewidth=2, color='#d62728', label="Bubble Cost")
plt.xlabel("Input Size N")
plt.ylabel("Cost (Time × Cores)")
plt.title("Bubble Sort: Parallel Execution Cost Analysis", fontsize=12, fontweight='bold')
plt.xticks(N, rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Calculate and display crossover point in console
idx = np.argmin(np.abs(bubble_seq - bubble_par))
print(f"Bubble crossover near N = {N[idx]}")