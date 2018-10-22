from carotid import carotid_data_util as cdu
from my_util import data_util
from sklearn.metrics import classification_report

dataset = 'done'
target = 'LEICA'
seed = 7
if dataset == 'done':
    id_all, x_data_all, y_data_all = cdu.get_done(target)
    fName = 'svm_new_done'
else:
    id_all, x_data_all, y_data_all = cdu.get_ko(target)
    fName = 'svm_ko'

predict = []
if target == 'RCCA':
    # RCCA:  XRCB_St  or  XRCD_St  or  XRCM_St  or  XRCP_St  >  50
    for index, row in x_data_all.iterrows():
        if ((row['XRCB_St'] >50.) | (row['XRCD_St'] > 50.) | (row['XRCM_St'] > 50.) | (row['XRCP_St'] > 50.)):
            predict.append(1)
        else:
            predict.append(0)
elif target == 'REICA':
    # REICA:  XRI_St  >  50  or  XRI_PS  >  125  or  XRI_PS/XRC_PS  >  2
    for index, row in x_data_all.iterrows():
        if ((row['XRI_St'] >50.) | (row['XRI_PS'] > 125.) | (row['XRI_PS']/row['XRC_PS'] > 2.)):
            predict.append(1)
        else:
            predict.append(0)
elif target == 'REVA':
    # REVA:  XRV_St  >  50
    for index, row in x_data_all.iterrows():
        if (row['XRV_St'] > 50.):
            predict.append(1)
        else:
            predict.append(0)
elif target == 'LCCA':
    # LCCA:  XLCB_St  or  XLCD_St  or  XLCM_ST  or  XLCP_St  >50
    for index, row in x_data_all.iterrows():
        if ((row['XLCB_St'] >50.) | (row['XLCD_St'] > 50.) | (row['XLCM_ST'] > 50.) | (row['XLCP_St'] > 50.)):
            predict.append(1)
        else:
            predict.append(0)
elif target == 'LEICA':
    # LEICA:  XLI_St  >  50  or  XLI_PS  >  125  or  XLI_PS/XLC_PS  >  2
    for index, row in x_data_all.iterrows():
        if ((row['XLI_St'] >50.) | (row['XLI_PS'] > 125.) | (row['XLI_PS']/row['XLC_PS'] > 2.)):
            predict.append(1)
        else:
            predict.append(0)
elif target == 'LEVA':
    # LEVA:  XLV_St  >  50
    for index, row in x_data_all.iterrows():
        if (row['XLV_St'] > 50.):
            predict.append(1)
        else:
            predict.append(0)
print(classification_report(y_data_all.values, predict, digits=3))