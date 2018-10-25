import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from carotid import carotid_data_util as cdu
import matplotlib.pyplot as plt

def plot_all_features(importances, feature_names):
    indices = np.argsort(importances)
    for feature in zip(feature_names, rf.feature_importances_):
        print(feature)
    plt.figure(1)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), feature_names[indices], fontsize=4)
    plt.xlabel('Relative Importance')
    plt.show()


seed = 7
target = 'RCCA'
soure = 'ex'


for i in range(0, 10):
    if(soure == 'exin'):
        id_all, x_data_all, y_data_all = cdu.get_exin_data(target)
        fName = 'fs_exin_'+target
    else:
        id_all, x_data_all, y_data_all = cdu.get_ex_data(target)
        fName = 'fs_ex_'+target

    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
    forest.fit(x_data_all, y_data_all)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(x_data_all.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(x_data_all.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(x_data_all.shape[1]), indices)
    plt.xlim([-1, x_data_all.shape[1]])
    plt.show()
