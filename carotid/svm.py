import sys
sys.path.append("..")
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from carotid import carotid_data_util as cdu
from my_util import data_util
import pandas as pd


def do_svn(target, soure, feature_selection):
    seed = 7
    # boost leave-one-out
    for i in range(0, 10):
        if(soure == 'exin'):
            if(feature_selection == 'fs'):
                id_all, x_data_all, y_data_all = cdu.get_exin_fs_data(target)
                fName = 'svm_exin_'+target+'_fs'
            else:
                id_all, x_data_all, y_data_all = cdu.get_exin_data(target)
                fName = 'svm_exin_'+target
        else:
            if(feature_selection == 'fs'):
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


if __name__ == '__main__':
    target = 'RMCA'
    source = 'ex'
    feature_selection = 'fs'
    do_svn(target, source, feature_selection)
    # do_svn(sys.argv[1], sys.argv[2], sys.argv[2])