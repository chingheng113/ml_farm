from carotid import carotid_data_util as cdu
import numpy as np
from my_util import data_util
from sklearn.metrics import classification_report, confusion_matrix

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

# RCCA, REICA, LCCA, LEICA
target = 'LEICA'
seed = 7
sen =[]
spe = []
acc = []
for i in range(0, 10):
    id_all, x_data_all, y_data_all = cdu.get_ex_data(target)
    predict = []
    if target == 'RCCA':
        # RCCA:  XRCB_St  or  XRCD_St  or  XRCM_St  or  XRCP_St  >  50
        for index, row in x_data_all.iterrows():
            if ((row['XRCB_St'] >50.) | (row['XRCD_St'] > 50.) | (row['XRCM_St'] > 50.) | (row['XRCP_St'] > 50.)):
                predict.append(1)
            else:
                predict.append(0)
    elif target == 'REICA':
        # REICA:  XRI_St  >  50  or  XRI_PS  >  125  or  XRI_PS/XRC_PS  >  2 or XRI_ED > 40
        for index, row in x_data_all.iterrows():
            if ((row['XRI_St'] >50.) | (row['XRI_PS'] > 125.) | (row['XRI_PS']/row['XRC_PS'] > 2.) | (row['XRI_ED'] > 40.)):
                predict.append(1)
            else:
                predict.append(0)
    # elif target == 'REVA':
    #     # REVA:  XRV_St  >  50
    #     for index, row in x_data_all.iterrows():
    #         if (row['XRV_St'] > 50.):
    #             predict.append(1)
    #         else:
    #             predict.append(0)
    elif target == 'LCCA':
        # LCCA:  XLCB_St  or  XLCD_St  or  XLCM_ST  or  XLCP_St  >50
        for index, row in x_data_all.iterrows():
            if ((row['XLCB_St'] >50.) | (row['XLCD_St'] > 50.) | (row['XLCM_ST'] > 50.) | (row['XLCP_St'] > 50.)):
                predict.append(1)
            else:
                predict.append(0)
    elif target == 'LEICA':
        # LEICA:  XLI_St  >  50  or  XLI_PS  >  125  or  XLI_PS/XLC_PS  >  2 or XLI_ED > 40
        for index, row in x_data_all.iterrows():
            if ((row['XLI_St'] >50.) | (row['XLI_PS'] > 125.) | (row['XLI_PS']/row['XLC_PS'] > 2.) | (row['XLI_ED'] > 40.)):
                predict.append(1)
            else:
                predict.append(0)
    # elif target == 'LEVA':
    #     # LEVA:  XLV_St  >  50
    #     for index, row in x_data_all.iterrows():
    #         if (row['XLV_St'] > 50.):
    #             predict.append(1)
    #         else:
    #             predict.append(0)
    print(classification_report(y_data_all.values, predict, digits=3))
    cm = confusion_matrix(y_data_all, predict)
    TPR, TNR, ACC = get_p(cm)
    sen.append(TPR)
    spe.append(TNR)
    acc.append(ACC)
print(target)
print(round(np.mean(acc), 4), round(np.std(acc), 4))
print(round(np.mean(sen), 4), round(np.std(sen), 4))
print(round(np.mean(spe), 4), round(np.std(spe), 4))