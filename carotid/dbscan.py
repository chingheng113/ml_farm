from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from my_util import data_util
from sklearn import metrics
from carotid import carotid_data_util as cdu
import numpy as np
import pandas as pd


target = 'RCCA'
seed = 7
id_all, x_data_all, y_data_all = cdu.get_ex_data(target)
fName = 'dbscan.csv'
x_data_all = data_util.scale(x_data_all)
labels_true = y_data_all.values.ravel()

# Compute DBSCAN
mSample = round(id_all.shape[0]/100, 0)
db = DBSCAN(eps=300, min_samples=mSample).fit(x_data_all)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
# Noise
labels = db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f"
#       % metrics.adjusted_rand_score(labels_true, labels))
# print("Adjusted Mutual Information: %0.3f"
#       % metrics.adjusted_mutual_info_score(labels_true, labels))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(x_data, labels))
#


# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 0]

    class_member_mask = (labels == k)

    xy = x_data_all[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)

    xy = x_data_all[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=2)

plt.title(target + '_DBSCAN: t-sne 2D\n Estimated number of clusters: %d' % n_clusters_)
plt.show()