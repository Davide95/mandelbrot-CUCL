import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

if len(sys.argv) < 2:
    logging.critical('Please specify the output file as a parameter.')
    exit(-1)

data = pd.read_csv(sys.argv[1], header=None, index_col=False).to_numpy()

rgb_image = np.zeros((2000, 3000, 3), dtype=np.uint8)

rgb_image[:, :, 0] = data*data * 255
rgb_image[:, :, 1] = data*data * 255
rgb_image[:, :, 2] = data * 255

plt.imshow(rgb_image)
plt.axis('off')
plt.show()
