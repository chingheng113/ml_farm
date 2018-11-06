import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from carotid import carotid_data_util as cdu
import matplotlib.pyplot as plt
import csv, os
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from my_util import data_util
from sklearn.metrics import classification_report


def plot_all_features(source, target, importances, feature_names):
    indices = np.argsort(importances)
    plt.figure(1)
    plt.title('10 bootstrap average feature importances:'+source+'_'+target)
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), feature_names[indices], fontsize=6)
    plt.xlabel('Relative Importance')
    plt.savefig('figures'+os.sep+target+'_'+source+'_fs.jpeg')
    # plt.show()


seed = 7
# targets = ['RCCA', 'REICA', 'RIICA', 'RACA', 'RMCA', 'RPCA', 'REVA', 'RIVA', 'BA', 'LCCA', 'LEICA', 'LIICA', 'LACA',
#            'LMCA', 'LPCA', 'LEVA', 'LIVA']
target = 'LMCA'
source = 'ex'
feature_names=''
all_importance = []
portions = [0.25, 0.5, 0.75]
predicts_25 = ['predict']
labeles_25 = ['label']
predicts_50 = ['predict']
labeles_50 = ['label']
predicts_75 = ['predict']
labeles_75 = ['label']
for i in range(0, 10):
    if(source == 'exin'):
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
    if i == 0:
        all_importance = importances
    else:
        all_importance = np.vstack((all_importance, importances))
    indices = np.argsort(importances)[::-1]
    # classifier =====
    for portion in portions:
        cut = int(round(len(indices)*portion, 0))
        indices_cut = indices[0:cut]
        x_data = x_data_all.ix[:, indices_cut]
        scaled_data = data_util.scale(x_data)
        x_data = pd.DataFrame(scaled_data, index=x_data.index, columns=x_data.columns)
        for train, test in LeaveOneOut().split(x_data_all):
            y_train = y_data_all.iloc[train]
            y_test = y_data_all.iloc[test]
            classifier = SVC(kernel='linear', random_state=seed, verbose=False)
            classifier.fit(x_data.iloc[train], y_train)
            predict = classifier.predict(x_data.iloc[test])
            label = y_data_all.iloc[test].values[0][0]
            if portion == 0.25:
                predicts_25.append(str(predict[0]))
                labeles_25.append(str(label))
            elif portion == 0.5:
                predicts_50.append(str(predict[0]))
                labeles_50.append(str(label))
            else:
                predicts_75.append(str(predict[0]))
                labeles_75.append(str(label))
# plot_all_features(source, target, np.mean(all_importance, axis=0), feature_names)
print(classification_report(labeles_25, predicts_25, digits=4))
r_25 = np.column_stack((labeles_25, predicts_25))
data_util.save_np_array_to_csv(r_25, target + '_' + source + '_25')
print(classification_report(labeles_50, predicts_50, digits=4))
r_50 = np.column_stack((labeles_50, predicts_50))
data_util.save_np_array_to_csv(r_50, target + '_' + source + '_50')
print(classification_report(labeles_75, predicts_75, digits=4))
r_75 = np.column_stack((labeles_75, predicts_75))
data_util.save_np_array_to_csv(r_75, target + '_' + source + '_75')
print("===")

