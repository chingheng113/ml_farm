from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,StratifiedKFold
from carotid import carotid_data_util as cdu
from my_util import data_util

dataset = 'done'
target = 'REVA'
seed = 7
if dataset == 'done':
    id_all, x_data_all, y_data_all = cdu.get_done(target)
    fName = 'svm_new_done'
else:
    id_all, x_data_all, y_data_all = cdu.get_ko(target)
    fName = 'svm_ko'
'''
id_train, id_test, x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(id_all, x_data_all, y_data_all, test_size=0.3, random_state=seed)
x_data_train = data_util.scale(x_data_train)
x_data_test = data_util.scale(x_data_test)

classifier = SVC(kernel='linear', probability=True, random_state=seed, verbose=True)
classifier.fit(x_data_train, y_data_train)

train_probas = classifier.predict_proba(x_data_test)
predict_result_hold = id_test
predict_result_hold['label'] = y_data_test
predict_result_hold['0'] = train_probas[:, 0]
predict_result_hold['1'] = train_probas[:, 1]
predict_result_hold.to_csv(cdu.get_save_path(fName+'.csv'), sep=',', encoding='utf-8')
'''

# 10-fold
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
for index, (train, test) in enumerate(kfold.split(x_data_all, y_data_all)):
    x_train = data_util.scale(x_data_all.iloc[train])
    x_test = data_util.scale(x_data_all.iloc[test])
    y_train = y_data_all.iloc[train]
    classifier = SVC(kernel='linear', probability=True, random_state=seed, verbose=True)
    classifier.fit(x_train, y_train)
    test_probas = classifier.predict_proba(x_test)
    predict_result_hold = id_all.iloc[test]
    predict_result_hold['label'] = y_data_all.iloc[test]
    predict_result_hold['0'] = test_probas[:, 0]
    predict_result_hold['1'] = test_probas[:, 1]
    predict_result_hold.to_csv(cdu.get_save_path(fName+'_'+str(index)+'.csv'), sep=',', encoding='utf-8')

