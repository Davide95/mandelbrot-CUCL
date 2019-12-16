import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  

plt.figure(1)
plt.boxplot([[195, 195, 195, 197, 196], \
             [130, 130, 130, 130, 130], \
             [120, 120, 120, 120, 120], \
             [120, 120, 120, 120, 120], \
             [105, 105, 105, 106, 106], \
             [125, 125, 125, 125, 125]])
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

# OpenMP + MPI
nn = [1, 2, 3, 4, 5, 6, 8, 9, 10]
ws = [48, 24, 29, 22, 22, 17, 11, 11, 9]
plt.figure(5)
plt.xlabel('Number of nodes')
plt.ylabel('Time (sec)')
plt.title('Strong scaling (RESOLUTION=3000, ITERATIONS=1000)')
plt.gca().set_xticks(nn)
plt.plot(nn, ws)
plt.show()

speedup = np.zeros_like(ws, dtype=np.float)
for idx, t in enumerate(ws):
    speedup[idx] = ws[0] / t

plt.figure(6)
plt.xlabel('Number of nodes')
plt.ylabel('Speedup')
plt.title('Strong scaling (RESOLUTION=3000, ITERATIONS=1000)')
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
plt.title('Strong scaling (RESOLUTION=3000, ITERATIONS=1000)')
plt.gca().set_xticks(nn)
plt.plot([1, 10], [1, 1], linestyle='--', color='gray')
plt.plot(nn, efficency)
plt.show()

# OpenMPI + MPI + degree=2
nn = [1, 2, 3, 4, 5, 6, 8, 9, 10]
ws = [79, 40, 31, 23, 19, 16, 12, 11, 10]
plt.figure(8)
plt.xlabel('Number of nodes')
plt.ylabel('Time (sec)')
plt.title('Strong scaling (RESOLUTION=12000, ITERATIONS=2000)')
plt.gca().set_xticks(nn)
plt.plot(nn, ws)
plt.show()

speedup = np.zeros_like(ws, dtype=np.float)
for idx, t in enumerate(ws):
    speedup[idx] = ws[0] / t

plt.figure(9)
plt.xlabel('Number of nodes')
plt.ylabel('Speedup')
plt.title('Strong scaling (RESOLUTION=12000, ITERATIONS=2000)')
plt.gca().set_xticks(nn)
plt.plot([1, 10], [1, 10], linestyle='--', color='gray')
plt.plot(nn, speedup)
plt.show()

efficency = np.zeros_like(speedup, dtype=np.float)
for idx, sp in enumerate(speedup):
    efficency[idx] = sp / nn[idx]

plt.figure(10)
plt.xlabel('Number of nodes')
plt.ylabel('Efficency')
plt.title('Strong scaling (RESOLUTION=12000, ITERATIONS=2000)')
plt.gca().set_xticks(nn)
plt.plot([1, 10], [1, 1], linestyle='--', color='gray')
plt.plot(nn, efficency)
plt.show()


data = np.array([[10, 11, 13, 15],
        [13, 15,18, 21],
        [16, 19, 22, 26],
        [20, 23, 27, 31]])
xlabels = np.array([12*12, 13*13, 14*14, 15*14])
ylabels = np.array([2, 3, 4, 5])

fig, ax = plt.subplots()
im = ax.imshow(data, cmap='GnBu')
ax.set_xticks(np.arange(len(xlabels)))
ax.set_xticklabels(xlabels)
ax.xaxis.tick_top()
ax.set_xlabel('Resolution (divided by 10^3)')
ax.xaxis.set_label_position('top')
ax.set_yticks(np.arange(len(ylabels)))
ax.set_yticklabels(ylabels)
ax.set_ylim(3.5, -0.5)
ax.set_ylabel('Iterations')
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Time (sec)', rotation=-90, va='bottom')
#ax.set_title(title)

for x_idx in range(len(xlabels)):
        for y_idx in range(len(ylabels)):
            ax.text(x_idx, y_idx, data[x_idx, y_idx],
                       ha="center", va="center", color="k")

fig.tight_layout()
plt.show()

plt.figure(12)
plt.boxplot([[154], \
             [152], \
             [152], \
             [152]])
plt.gca().set_xticklabels(['-O0', '-O1', '-O2', '-O3'])
plt.ylabel('Time (sec)')
plt.title('Comparison of different compiler flags (RESOLUTION=12000, ITERATIONS=2000)')
plt.show()

plt.figure(13)
plt.boxplot([[152],
             [42]])
plt.gca().set_xticklabels(['cucomplex.cu', 'vanilla.cu'])
plt.ylabel('Time (sec)')
plt.title('Comparison of different implementations (RESOLUTION=12000, ITERATIONS=2000)')
plt.show()


# Multiple GPUs
nn = [1, 2, 4]
ws = [42, 22, 18]
plt.figure(8)
plt.xlabel('Number of GPUs')
plt.ylabel('Time (sec)')
plt.title('Strong scaling (RESOLUTION=12000, ITERATIONS=2000)')
plt.gca().set_xticks(nn)
plt.plot(nn, ws)
plt.show()

speedup = np.zeros_like(ws, dtype=np.float)
for idx, t in enumerate(ws):
    speedup[idx] = ws[0] / t

plt.figure(14)
plt.xlabel('Number of GPUs')
plt.ylabel('Speedup')
plt.title('Strong scaling (RESOLUTION=12000, ITERATIONS=2000)')
plt.gca().set_xticks(nn)
plt.plot([1, 4], [1, 4], linestyle='--', color='gray')
plt.plot(nn, speedup)
plt.show()

efficency = np.zeros_like(speedup, dtype=np.float)
for idx, sp in enumerate(speedup):
    efficency[idx] = sp / nn[idx]

plt.figure(15)
plt.xlabel('Number of GPUs')
plt.ylabel('Efficency')
plt.title('Strong scaling (RESOLUTION=12000, ITERATIONS=2000)')
plt.gca().set_xticks(nn)
plt.plot([1, 4], [1, 1], linestyle='--', color='gray')
plt.plot(nn, efficency)
plt.show()
