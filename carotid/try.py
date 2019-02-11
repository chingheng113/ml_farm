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
colors = cm.gnuplot(np.linspace(0, 0.5, len(targets)))
fig, axes = plt.subplots(nrows=17, ncols=2, sharex=True, figsize=(10, 20), constrained_layout=False)
# if source == 'ex':
#     plt.suptitle('Sensitive changes of different number of extracranial inputs')
# else:
#     plt.suptitle('Sensitive changes of different number of extracranial and intracranial inputs')
for inx, target in enumerate(targets):
    sen_por = []
    sdv_por = []
    for portion in portions:
        result = cdu.get_result('fs_grid'+os.sep+target+'_ex_'+str(portion)+'_fs.csv')
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
        print(target, portion, round(np.mean(sen), 3), round(np.std(sen), 3))
        sen_por.append(round(np.mean(sen), 3))
        sdv_por.append(round(np.std(sen), 3))

    axes[inx,0].errorbar(portions, sen_por, yerr=sdv_por, label=target, c=colors[inx])
    axes[inx,0].set_facecolor('silver')
    axes[inx,0].set_xlim(0.2, 0.7)
axes[0,0].title.set_text('Extracranial Inputs')

for inx, target in enumerate(targets):
    sen_por = []
    sdv_por = []
    for portion in portions:
        result = cdu.get_result('fs_grid'+os.sep+target+'_exin_'+str(portion)+'_fs.csv')
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
        print(target, portion, round(np.mean(sen), 3), round(np.std(sen), 3))
        sen_por.append(round(np.mean(sen), 3))
        sdv_por.append(round(np.std(sen), 3))

    axes[inx,1].errorbar(portions, sen_por, yerr=sdv_por, label=target, c=colors[inx])
    axes[inx,1].set_facecolor('silver')
    axes[inx,1].legend(loc="center left", bbox_to_anchor=(1, 0.5), prop={'size': 8})
    axes[inx,1].set_xlim(0.2, 0.7)
axes[0,1].title.set_text('Extracranial and Intracranial Inputs')

fig.text(0.5, 0.05, 'Proportion of selection', ha='center', va='center')
fig.text(0.03, 0.5, 'Sensitivity', ha='center', va='center', rotation='vertical')

plt.subplots_adjust(hspace=0.5)
plt.show()
print('done')


