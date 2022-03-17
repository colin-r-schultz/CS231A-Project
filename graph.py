import numpy as np
import matplotlib.pyplot as plt

def get_median(fname):
    x = np.loadtxt(fname, delimiter=",")
    return np.median(x, axis=1)

rm = get_median("random_results.txt")
cm = get_median("cont_results.txt")
fm = get_median("features_results.txt")
sm = get_median("spatial_segmentation_results.txt")
# smm = get_median("split_and_merge_results.txt")

x = range(2, 9)

plt.gca().set_title("Cluster quality by number of objects")
# plt.plot(x, rm, label="Random baseline")
# plt.plot(x, cm, label="Direct Optimization")
# plt.plot(x, fm, label="Gradient Features")
plt.plot(x, sm, label="Spatial Clustering")
# plt.plot(x, smm, label="Split-and-Merge")
plt.gca().set_ylim([0, 1])
plt.gca().set_xlabel("Number of objects")
plt.gca().set_ylabel("V-measure of segmentation")
plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()