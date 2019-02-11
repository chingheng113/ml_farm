from sklearn.neighbors import NearestNeighbors
from carotid import carotid_data_util as cdu
from my_util import data_util
import matplotlib.pyplot as plt


id_all, x_data_all, y_data_all = cdu.get_ex_normal()
# id_all, x_data_all, y_data_all = cdu.get_ex_data('RCCA')
# x_data_all = data_util.scale(x_data_all)
# a minimum minPts can be derived from the number of dimensions D in the data set, as minPts ≥ D + 1
minPts = x_data_all.shape[1]+1
# The value for ε can then be chosen by using a k-distance graph, plotting the distance to the k = minPts-1
k = minPts-1
nbrs = NearestNeighbors(n_neighbors=k).fit(x_data_all)
distances, indices = nbrs.kneighbors(x_data_all)
distanceDec = sorted(distances[:,k-1], reverse=True)
plt.plot(list(range(1, x_data_all.shape[0]+1)), distanceDec)
plt.show()

