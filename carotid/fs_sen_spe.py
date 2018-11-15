from carotid import carotid_data_util as cdu
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

targets = ['RCCA', 'REICA', 'RIICA', 'RACA', 'RMCA', 'RPCA', 'REVA', 'RIVA', 'BA', 'LCCA', 'LEICA', 'LIICA', 'LACA',
           'LMCA', 'LPCA', 'LEVA', 'LIVA']
source = 'ex'
portions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
colors = cm.gnuplot(np.linspace(0, 1, len(targets)))
f, axes = plt.subplots(17, 1, sharex='col', sharey='row', figsize=(8, 8), constrained_layout=True)
if source == 'exin':
    plt.suptitle('Sensitive changes of different number of extracranial inputs')
else:
    plt.suptitle('Sensitive changes of different number of extracranial and intracranial inputs')
for inx, target in enumerate(targets):
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
        print(target, portion, round(np.mean(sen), 4), round(np.std(sen), 4))
        sen_por.append(round(np.mean(sen), 4))
        sdv_por.append(round(np.std(sen), 4))

    axes[inx].errorbar(portions, sen_por, yerr=sdv_por, label=target, c=colors[inx])
    axes[inx].set_facecolor('silver')
    axes[inx].legend(loc="center left", bbox_to_anchor=(1, 0.5), prop={'size': 8})
    axes[inx].set_xlim(0.2, 0.7)
    # plt.errorbar(portions, sen_por, yerr=sdv_por, label=target, c=colors[inx])
    # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), prop={'size': 8})
plt.show()
print('done')


