import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from carotid import carotid_data_util as cdu
import matplotlib.pyplot as plt
import csv, os


def plot_all_features(source, target, importances, feature_names):
    indices = np.argsort(importances)
    plt.figure(1)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), feature_names[indices], fontsize=6)
    plt.xlabel('Relative Importance')
    # plt.savefig('figures'+os.sep+target+'_'+source+'_fs.jpeg')
    plt.show()


seed = 7
targets = ['RCCA', 'REICA', 'RIICA', 'RACA', 'RMCA', 'RPCA', 'REVA', 'RIVA', 'BA', 'LCCA', 'LEICA', 'LIICA', 'LACA',
           'LMCA', 'LPCA', 'LEVA', 'LIVA']
soure = 'exin'
for index, target in enumerate(targets):
    with open('fs'+os.sep+target+'_'+soure+'_fs.csv', 'w', newline="") as csv_file:
        for i in range(0, 10):
            if(soure == 'exin'):
                id_all, x_data_all, y_data_all = cdu.get_exin_data(target)
                fName = 'fs_exin_'+target
            else:
                id_all, x_data_all, y_data_all = cdu.get_ex_data(target)
                fName = 'fs_ex_'+target
            feature_names = x_data_all.columns
            # Build a forest and compute the feature importances
            forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
            forest.fit(x_data_all, y_data_all)
            importances = forest.feature_importances_
            # plot_all_features(soure, target, importances, feature_names)
            indices = np.argsort(importances)[::-1]
            selected_f = []
            for f in range(x_data_all.shape[1]):
                if importances[indices[f]] > 1e-2:
                    fn = feature_names[indices[f]]
                    im = importances[indices[f]]
                    # print(fn, im)
                    selected_f.append(fn)
            wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
            wr.writerow(selected_f)
print("===")

