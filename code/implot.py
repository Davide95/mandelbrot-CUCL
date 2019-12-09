import sys, logging, warnings
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math

if len(sys.argv) < 2:
    logging.critical('Please specify the input file as a parameter.')
    exit(-1)

data = pd.read_csv(sys.argv[1], header=None, index_col=False).to_numpy()

# If np.log(0), it won't color the pixel. That's great.
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    data = np.log(data / np.max(data))

plt.imshow(data, cmap='gist_rainbow')
plt.axis('off')
plt.show()
