from my_util import data_util
from carotid import carotid_data_util as cdu
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import os
from scipy import interp

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


def calculate_roc_auc(label, probas_):
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(label, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc
'''
RCCA
REICA
RIICA
RACA
RMCA
RPCA
REVA
RIVA
BA
LCCA
LEICA
LIICA
LACA
LMCA
LPCA
LEVA
LIVA
'''
target = 'LEVA'
source = 'exin'
classifier = 'svm'
feature_selection = False
# result = cdu.get_result(classifier+'_'+soure+'_'+target+'.csv')
# label = result['label']
# probas_ = result[['0', '1']].values
# predict = data_util.labelize(probas_)
# print(confusion_matrix(label, predict))
# print('Note that in binary classification, recall of the positive class is also known as “sensitivity”; recall of the negative class is “specificity”.')
# print(classification_report(label, predict, digits=4))
# fpr, tpr, thresholds = roc_curve(label, probas_[:, 1])
# roc_auc = auc(fpr, tpr)
# plt.plot(fpr, tpr,
#          label=' (AUC = %0.3f )' % roc_auc,
#          lw=1, alpha=.8)
# plt.show()
sen =[]
spe = []
acc = []
auc_arr = []
for inx in range(0, 10, 1):
    if feature_selection:
        result = cdu.get_result(source+'_fs'+os.sep+classifier+'_'+source+'_'+target+'_fs_'+str(inx)+'.csv')
    else:
        result = cdu.get_result(source+os.sep+classifier+'_'+source+'_'+target+'_'+str(inx)+'.csv')
        # result = cdu.get_result(classifier+'_'+source+'_'+target+'_'+str(inx)+'.csv')
    label = list(result['label'].values)
    probas_ = result[['0', '1']].values
    predict = list(data_util.labelize(probas_))
    print(classification_report(label, predict, digits=4))
    cm = confusion_matrix(label, predict)
    TPR, TNR, ACC = get_p(cm)
    fpr, tpr, roc_auc = calculate_roc_auc(label, probas_)
    if inx == 0:
        labels = label
        predicts = predict
    else:
        labels.extend(label)
        predicts.extend(predict)
    sen.append(TPR)
    spe.append(TNR)
    acc.append(ACC)
    auc_arr.append(roc_auc)
print('---')
print(classification_report(labels, predicts, digits=4))
# print(round(accuracy_score(labels, predicts), 4))
print(target)
print(round(np.mean(acc)*100, 2), round(np.std(acc)*100, 2))
print(round(np.mean(sen)*100, 2), round(np.std(sen)*100, 2))
print(round(np.mean(spe)*100, 2), round(np.std(spe)*100, 2))
print(round(np.mean(auc_arr)*100, 2), round(np.std(auc_arr)*100, 2))
