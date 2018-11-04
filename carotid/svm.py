from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from carotid import carotid_data_util as cdu
from my_util import data_util
import pandas as pd

seed = 7
target = 'BA'
soure = 'ex'
feature_selection = True

# hold-out
# if(soure == 'exin'):
#     id_all, x_data_all, y_data_all = cdu.get_exin_data(target)
#     fName = 'svm_exin_'+target
# else:
#     id_all, x_data_all, y_data_all = cdu.get_ex_data(target)
#     fName = 'svm_ex_'+target
# id_train, id_test, x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(id_all, x_data_all, y_data_all, test_size=0.3, random_state=seed)
# x_data_train = data_util.scale(x_data_train)
# x_data_test = data_util.scale(x_data_test)
#
# classifier = SVC(kernel='linear', probability=True, random_state=seed, verbose=True)
# classifier.fit(x_data_train, y_data_train)
#
# train_probas = classifier.predict_proba(x_data_test)
# predict_result_hold = id_test
# predict_result_hold['label'] = y_data_test
# predict_result_hold['0'] = train_probas[:, 0]
# predict_result_hold['1'] = train_probas[:, 1]
# predict_result_hold.to_csv(cdu.get_save_path(fName+'.csv'), sep=',', encoding='utf-8')


# 10-fold
# if(soure == 'exin'):
#     id_all, x_data_all, y_data_all = cdu.get_exin_data(target)
#     fName = 'svm_exin_'+target
# else:
#     id_all, x_data_all, y_data_all = cdu.get_ex_data(target)
#     fName = 'svm_ex_'+target
# kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
# for index, (train, test) in enumerate(kfold.split(x_data_all, y_data_all)):
#     x_train = data_util.scale(x_data_all.iloc[train])
#     x_test = data_util.scale(x_data_all.iloc[test])
#     y_train = y_data_all.iloc[train]
#     classifier = SVC(kernel='linear', probability=True, random_state=seed, verbose=True)
#     classifier.fit(x_train, y_train)
#     test_probas = classifier.predict_proba(x_test)
#     predict_result_hold = id_all.iloc[test]
#     predict_result_hold['label'] = y_data_all.iloc[test]
#     predict_result_hold['0'] = test_probas[:, 0]
#     predict_result_hold['1'] = test_probas[:, 1]
#     predict_result_hold.to_csv(cdu.get_save_path(fName+'_'+str(index)+'.csv'), sep=',', encoding='utf-8')

# boost leave-one-out
for i in range(0, 10):
    if(soure == 'exin'):
        if(feature_selection):
            id_all, x_data_all, y_data_all = cdu.get_exin_fs_data(target)
            fName = 'svm_exin_'+target+'_fs'
        else:
            id_all, x_data_all, y_data_all = cdu.get_exin_data(target)
            fName = 'svm_exin_'+target
    else:
        if(feature_selection):
            id_all, x_data_all, y_data_all = cdu.get_ex_fs_data(target)
            fName = 'svm_ex_'+target+'_fs'
        else:
            id_all, x_data_all, y_data_all = cdu.get_ex_data(target)
            fName = 'svm_ex_'+target
    lst = []
    scaled_data = data_util.scale(x_data_all)
    x_data_all = pd.DataFrame(scaled_data, index=x_data_all.index, columns=x_data_all.columns)
    for train, test in LeaveOneOut().split(x_data_all):
        y_train = y_data_all.iloc[train]
        classifier = SVC(kernel='linear', probability=True, random_state=seed, verbose=True)
        classifier.fit(x_data_all.iloc[train], y_train)
        test_probas = classifier.predict_proba(x_data_all.iloc[test])
        one_reslut = test_probas[0]
        lst.append([id_all.iloc[test].values[0][0], y_data_all.iloc[test].values[0][0], one_reslut[0], one_reslut[1]])
    predict_result = pd.DataFrame(lst, columns=['id', 'label', '0', '1'])
    predict_result.to_csv(cdu.get_save_path(fName+'_'+str(i)+'.csv'), sep=',', encoding='utf-8')

print('done')
