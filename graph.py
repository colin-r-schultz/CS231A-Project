import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def get_median(fname):
    x = np.loadtxt(fname, delimiter=",")
    return np.median(x, axis=1)

rm = get_median("random.txt")
cm = get_median("direct.txt")
fm = get_median("grad.txt")
sm = get_median("spatial.txt")
smm = get_median("split_merge.txt")
for m, l in zip([rm, cm, fm, sm, smm], ["r", "c", "f", "s", "sm"]):
    print(l)
    print(m[-1])

# x = 2**np.arange(1, 8)
x = range(2, 9)
plt.gca().set_title("Cluster quality by number of objects")
plt.plot(x, rm, label="Random baseline")
plt.plot(x, cm, label="Direct Optimization")
plt.plot(x, fm, label="Gradient Features")
plt.plot(x, sm, label="Spatial Clustering")
# plt.plot(x, smm, label="Split-and-Merge")
# plt.gca().set_xscale('log', base=2)
plt.gca().xaxis.set_major_formatter(FormatStrFormatter("%d"))
plt.gca().set_ylim([0, 1])
plt.gca().set_xlabel("Number of objects")
plt.gca().set_ylabel("V-measure of segmentation")
plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()