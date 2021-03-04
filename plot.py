import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

NX = 50
NY = 50
ch = 3
frames = 50

f = open("result", 'rb')
data = np.fromfile(f, dtype=np.float64)
data.shape = (frames, ch, NX, NY)

fig, ax = plt.subplots(ncols = 3)

sns.heatmap(data[49,0,:,:], ax=ax[0], xticklabels=False, yticklabels=False, cmap="viridis")
sns.heatmap(data[49,1,:,:], ax=ax[1], xticklabels=False, yticklabels=False, cmap="viridis")
sns.heatmap(data[49,2,:,:], ax=ax[2], xticklabels=False, yticklabels=False, cmap="viridis")

plt.show()
