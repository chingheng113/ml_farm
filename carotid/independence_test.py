from sklearn.feature_selection import mutual_info_regression
from carotid import carotid_data_util as cdu
from scipy.stats import pearsonr


dataset = 'ko'
target = 'Stenosis_code'
seed = 7
if dataset == 'ko':
    id_all, x_data_all, y_data_all = cdu.get_ko(target)
    fName = 'svm_ko.csv'
elif dataset == 'jim':
    id_all, x_data_all, y_data_all = cdu.get_jim(target)
    fName = 'svm_jim.csv'
elif dataset == 'jim2':
    id_all, x_data_all, y_data_all = cdu.get_jim2(target)
    fName = 'svm_jim2.csv'
else:
    id_all, x_data_all, y_data_all = cdu.get_new(target)
    fName = 'svm_new.csv'

x = x_data_all['XLC_PS'].values
y = x_data_all['XLC_ED'].values

print("mutal info", mutual_info_regression(x.reshape(-1, 1), y.reshape(-1, 1)))
print("Pearson (sorce, p-value)", pearsonr(x, y))
