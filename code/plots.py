import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  

plt.figure(1)
plt.boxplot([[195, 195, 195, 197, 196], \
             [130, 130, 130, 130, 130], \
             [120, 120, 120, 120, 120], \
             [120, 120, 120, 120, 120], \
             [105, 105, 105, 106, 106], \
             [125, 125, 125, 125, 125]]) # Segnare parametri usati
plt.gca().set_xticklabels(['-O0', '-O1', '-O2', '-O3', '-Ofast', '-Ofast -xHost'])
plt.ylabel('Time (sec)')
plt.title('Comparison of different compiler flags (RESOLUTION=500, ITERATIONS=500)')
plt.show()

plt.figure(2)
plt.boxplot([[16, 16, 16, 16, 16], \
             [105, 105, 105, 106, 106]])
plt.gca().set_xticklabels(['Consumer-level', 'Node of the cluster'])
plt.ylabel('Time (sec)')
plt.title('Comparison of different CPUs (RESOLUTION=500, ITERATIONS=500)')
plt.show()

plt.figure(3)
plt.boxplot([[78, 78, 78, 78, 78], \
             [51, 51, 51, 51, 51],
             [50, 50, 50, 50, 51]], whis=[0, 100])
plt.gca().set_xticklabels(['Static', 'Dynamic', 'Guided'])
plt.ylabel('Time (sec)')
plt.title('Comparison of different schedulers (RESOLUTION=3000, ITERATIONS=1000)')
plt.show()

plt.figure(4)
plt.boxplot([[51, 51, 51, 51, 51], \
             [50, 50, 50, 50, 50],
             [48, 48, 48, 48, 48]])
plt.gca().set_xticklabels(['Dynamic', 'Dynamic + seq. part', 'Dynamic + seq. part + -xHost'])
plt.ylabel('Time (sec)')
plt.title('Comparison (RESOLUTION=3000, ITERATIONS=1000)')
plt.show()

nn = [1, 2, 3, 4, 5, 6, 8, 9, 10]
ws = [48, 24, 29, 22, 22, 17, 11, 11, 9]
plt.figure(5)
plt.xlabel('Number of nodes')
plt.ylabel('Time (sec)')
plt.title('Weak scaling (RESOLUTION=3000, ITERATIONS=1000)')
plt.gca().set_xticks(nn)
plt.plot(nn, ws)
plt.show()

speedup = np.zeros_like(ws, dtype=np.float)
for idx, t in enumerate(ws):
    speedup[idx] = ws[0] / t

plt.figure(6)
plt.xlabel('Number of nodes')
plt.ylabel('Speedup')
plt.title('Weak scaling (RESOLUTION=3000, ITERATIONS=1000)')
plt.gca().set_xticks(nn)
plt.plot([1, 10], [1, 10], linestyle='--', color='gray')
plt.plot(nn, speedup)
plt.show()

efficency = np.zeros_like(speedup, dtype=np.float)
for idx, sp in enumerate(speedup):
    efficency[idx] = sp / nn[idx]

plt.figure(7)
plt.xlabel('Number of nodes')
plt.ylabel('Efficency')
plt.title('Weak scaling (RESOLUTION=3000, ITERATIONS=1000)')
plt.gca().set_xticks(nn)
plt.plot([1, 10], [1, 1], linestyle='--', color='gray')
plt.plot(nn, efficency)
plt.show()