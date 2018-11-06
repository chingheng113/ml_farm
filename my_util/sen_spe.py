from carotid import carotid_data_util as cdu
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


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


target ='LMCA'
source = 'ex'
result_25 = cdu.get_result(target+'_'+source+'_25.csv')
cm = confusion_matrix(result_25.iloc[:,0], result_25.iloc[:,1])
TPR, TNR, ACC = get_p(cm)
print(TPR, TNR, ACC)
print(classification_report(result_25.iloc[:,0], result_25.iloc[:,1], digits=4))

result_50 = cdu.get_result(target+'_'+source+'_50.csv')
cm = confusion_matrix(result_50.iloc[:,0], result_50.iloc[:,1])
TPR, TNR, ACC = get_p(cm)
print(TPR, TNR, ACC)
print(classification_report(result_50.iloc[:,0], result_50.iloc[:,1], digits=4))

result_75 = cdu.get_result(target+'_'+source+'_75.csv')
cm = confusion_matrix(result_75.iloc[:,0], result_75.iloc[:,1])
TPR, TNR, ACC = get_p(cm)
print(TPR, TNR, ACC)
print(classification_report(result_75.iloc[:,0], result_75.iloc[:,1], digits=4))


