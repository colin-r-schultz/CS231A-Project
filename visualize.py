import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

no_iters = np.load("features2objs0iters.npy")
iters = np.load("features2objs10000iters.npy")

tsne = TSNE(perplexity=2)

x = tsne.fit_transform(iters)
plt.scatter(x[:8, 0], x[:8, 1])
plt.scatter(x[8:, 0], x[8:, 1])
plt.show()