import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv("la1.txt")

N = df["N"].values

# ================= BFS =================
bfs_seq = df["SEQ"].values
bfs_par = df["PAR"].values
bfs_speed = df["SPEEDUP"].values

# CORES
CORES = 4
bfs_eff = bfs_speed / CORES

# ================= BFS TIME & CROSSOVER GRAPH =================
plt.figure()
plt.plot(N, bfs_seq, marker='o', label="BFS Sequential")
plt.plot(N, bfs_par, marker='o', label="BFS Parallel")
plt.xlabel("Input Size N")
plt.ylabel("Time (seconds)")
plt.title("BFS: Sequential vs Parallel\n(Crossover Detection)", fontsize=12)
plt.xticks(N, rotation=90)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ================= BFS SPEEDUP =================
plt.figure()
plt.plot(N, bfs_speed, marker='o', label="BFS Speedup")
plt.axhline(y=1, color='r', linestyle='--', label="No Speedup (1.0)")
plt.xlabel("Input Size N")
plt.ylabel("Speedup")
plt.title("BFS Speedup")
plt.xticks(N, rotation=90)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ================= BFS EFFICIENCY =================
plt.figure()
plt.plot(N, bfs_eff, marker='o', label="BFS Efficiency")
plt.axhline(y=1, color='r', linestyle='--', label="Ideal (1.0)")
plt.xlabel("Input Size N")
plt.ylabel("Efficiency")
plt.title("BFS Efficiency")
plt.xticks(N, rotation=90)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ================= COST GRAPH =================
bfs_cost = df["COST"].values
plt.figure()
plt.plot(N, bfs_cost, marker='o', label="BFS Cost", color='purple')
plt.xlabel("Input Size N")
plt.ylabel("Cost")
plt.title("BFS Cost Analysis")
plt.xticks(N, rotation=90)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


#  FIND CROSSOVER POINT
idx = np.argmin(np.abs(bfs_seq - bfs_par))
print("Crossover near N =", N[idx])