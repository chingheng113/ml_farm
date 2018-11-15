import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from carotid import carotid_data_util as cdu
import matplotlib.pyplot as plt
import csv, os


def plot_all_features(source, target, importances, feature_names):
    indices = np.argsort(importances)
    plt.figure(1)
    plt.title('10 bootstrap average feature importances:'+source+'_'+target)
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), feature_names[indices], fontsize=6)
    plt.xlabel('Relative Importance')
    plt.savefig('figures'+os.sep+target+'_'+source+'_fs.jpeg')
    # plt.show()


def get_p(cm):
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    return TPR, TNR, ACC


def highest_portion(target, source):
    portions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    sen_por = []
    sdv_por = []
    for portion in portions:
        result = cdu.get_result('fs_grid'+os.sep+target+'_'+source+'_'+str(portion)+'_fs.csv')
        length = int(result.shape[0]/10)
        start = 0
        end = length
        sen = []
        for i in range(0, 10):
            result_10 = result.iloc[start:end, :]
            cm = confusion_matrix(result_10['label'], result_10['predict'])
            TPR, TNR, ACC = get_p(cm)
            sen.append(TPR)
            # print(portion, TPR, TNR, ACC)
            start = length
            end = end+length
        # print(target, portion, round(np.mean(sen), 4), round(np.std(sen), 4))
        sen_por.append(round(np.mean(sen), 4))
    return portions[np.argmax(sen_por)]


seed = 7
targets = ['RCCA', 'REICA', 'RIICA', 'RACA', 'RMCA', 'RPCA', 'REVA', 'RIVA', 'BA', 'LCCA', 'LEICA', 'LIICA', 'LACA',
           'LMCA', 'LPCA', 'LEVA', 'LIVA']
# target = 'BA'
source = 'exin'
for target in targets:
    with open('fs'+os.sep+target+'_'+source+'_fs.csv', 'w', newline="") as csv_file:
        if(source == 'exin'):
            id_all, x_data_all, y_data_all = cdu.get_exin_data(target)
            fName = 'fs_exin_'+target
        else:
            id_all, x_data_all, y_data_all = cdu.get_ex_data(target)
            fName = 'fs_ex_'+target
        feature_names = x_data_all.columns.values
        # Build a forest and compute the feature importances
        forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
        forest.fit(x_data_all, y_data_all)
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]
        p = highest_portion(target, source)
        slice_indice = indices[0:int(indices.size*p)]
        selected_f = feature_names[slice_indice]
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        wr.writerow(selected_f)
    print("===")

