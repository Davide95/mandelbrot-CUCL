import matplotlib.pyplot as plt

plt.figure(1)
plt.boxplot([[195, 195, 195, 197, 196], \
             [130, 130, 130, 130, 130], \
             [120, 120, 120, 120, 120], \
             [120, 120, 120, 120, 120], \
             [105, 105, 105, 106, 106], \
             [125, 125, 125, 125, 125]])
plt.gca().set_xticklabels(['-O0', '-O1', '-O2', '-O3', '-Ofast', '-Ofast -xHost'])
plt.ylabel('Time (sec)')
plt.title('Comparison of different compiler flags')
plt.show()