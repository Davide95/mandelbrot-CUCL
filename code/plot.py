import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    logging.critical('Please specify the output file as a parameter.')
    exit(-1)

data = pd.read_csv(sys.argv[1], header=None, index_col=False).to_numpy()
plt.imshow(data)
plt.axis('off')
plt.show()