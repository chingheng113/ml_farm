from my_util import data_util
from carotid import carotid_data_util as cdu
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import os


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
target = 'RMCA'
source = 'ex'
classifier = 'cnn'

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
# for inx in range(0, 10, 1):
for inx in range(0, 3, 1):
    # result = cdu.get_result(source+os.sep+classifier+'_'+source+'_'+target+'_'+str(inx)+'.csv')
    result = cdu.get_result(classifier+'_'+source+'_'+target+'_'+str(inx)+'.csv')
    label = list(result['label'].values)
    probas_ = result[['0', '1']].values
    predict = list(data_util.labelize(probas_))
    print(classification_report(label, predict, digits=4))
    cm = confusion_matrix(label, predict)
    TPR, TNR, ACC = get_p(cm)
    if inx == 0:
        labels = label
        predicts = predict
    else:
        labels.extend(label)
        predicts.extend(predict)
    sen.append(TPR)
    spe.append(TNR)
    acc.append(ACC)

print('---')
print(classification_report(labels, predicts, digits=4))
# print(round(accuracy_score(labels, predicts), 4))
print(target)
print(round(np.mean(acc), 4), round(np.std(acc), 4))
print(round(np.mean(sen), 4), round(np.std(sen), 4))
print(round(np.mean(spe), 4), round(np.std(spe), 4))

