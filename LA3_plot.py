import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ================= LOAD DATA =================
df = pd.read_csv("reduction_result.csv")

# Average multiple runs
df = df.groupby("N").mean().reset_index()

# Sort values
df = df.sort_values("N")

N = df["N"].values

# ================= TIME =================
plt.figure()

plt.plot(N, df["SEQ_MS"], marker='o', label="Sequential")
plt.plot(N, df["PAR_MS"], marker='o', label="Parallel")

plt.xlabel("Input Size N")
plt.ylabel("Time (ms)")
plt.title("Crossover Graph (Execution Time)")

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ================= SPEEDUP =================
plt.figure()

plt.plot(N, df["SPEEDUP"], marker='o', label="Speedup")
plt.axhline(y=1, linestyle='--')

plt.xlabel("Input Size N")
plt.ylabel("Speedup")
plt.title("Speedup vs N")

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ================= EFFICIENCY =================
plt.figure()

plt.plot(N, df["EFFICIENCY"], marker='o', label="Efficiency")

plt.xlabel("Input Size N")
plt.ylabel("Efficiency")
plt.title("Efficiency vs N")

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ================= COST =================
plt.figure()

plt.plot(N, df["COST"], marker='o', label="Cost")

plt.xlabel("Input Size N")
plt.ylabel("Cost (time × cores)")
plt.title("Cost vs N")

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()