import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("la4.txt")
df = df.sort_values("N")

# ================= TIME =================
plt.figure()
plt.plot(df['N'], df['SEQ'], 'b*-', label='Sequential')
plt.plot(df['N'], df['PAR'], 'g*-', label='Parallel')

plt.xlabel("Input Size N")
plt.ylabel("Time (ms)")
plt.title("CPU vs GPU (Cross-over expected)")

plt.xticks(df['N'], rotation=45)   # normal values on x-axis

plt.legend()
plt.grid(True)
plt.show()

# ================= SPEEDUP =================
plt.figure()
plt.plot(df['N'], df['SPEEDUP'], 'r*-')

plt.axhline(y=1, linestyle='--')

plt.xlabel("Input Size N")
plt.ylabel("Speedup")
plt.title("Speedup increases with N")

plt.xticks(df['N'], rotation=45)

plt.grid(True)
plt.show()

# ================= EFFICIENCY =================
plt.figure()
plt.plot(df['N'], df['EFFICIENCY'], 'm*-')

plt.xlabel("Input Size N")
plt.ylabel("Efficiency")
plt.title("Efficiency Increasing Trend")

plt.xticks(df['N'], rotation=45)

plt.grid(True)
plt.show()

# ================= COST =================
plt.figure()
plt.plot(df['N'], df['COST'], 'c*-')

plt.xlabel("Input Size N")
plt.ylabel("Cost")
plt.title("Parallel Cost")

plt.xticks(df['N'], rotation=45)

plt.grid(True)
plt.show()