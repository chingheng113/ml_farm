from sklearn import preprocessing as sp
from sklearn.utils import resample
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
import os


def get_file_path(file_name):
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, '..' + os.sep + 'data_source' + os.sep)
    return os.path.join(filepath + file_name)


def load_all(fn):
    read_file_path = get_file_path(fn)
    df = pd.read_csv(read_file_path, encoding='utf8')
    # df = df.sample(frac=1)
    # df = df.ix[:10]
    return df


def variance_threshold_selector(data, threshold=0.5):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]


def scale(x_data):
    # x_data = np.round(sp.MinMaxScaler(feature_range=(0, 1)).fit_transform(x_data), 3)
    x_data = np.round(sp.StandardScaler().fit_transform(x_data), 3)
    return x_data


def save_dataframe_to_csv(df, file_name):
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, '..'+os.sep+'data_source'+os.sep)
    df.to_csv(filepath + file_name + '.csv', sep=',', index=False)


def save_np_array_to_csv(array, file_name):
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, '..'+os.sep+'data_source'+os.sep)
    np.savetxt(filepath + file_name + '.csv', array, delimiter=',', fmt='%s')


def labelize(y_arr):
    y_label = []
    for y in y_arr:
        y_label = np.append(y_label, np.argmax(y))
    return y_label


cnn_col_ed = ['XLC_ED', 'XLE_ED', 'XLI_ED', 'XLS_ED', 'XLV_ED', 'XRC_ED', 'XRE_ED', 'XRI_ED', 'XRS_ED', 'XRV_ED']
cnn_col_fv = ['XLC_FV', 'XLE_FV', 'XLI_FV', 'XLV_FV', 'XRC_FV', 'XRE_FV', 'XRI_FV', 'XRV_FV']
cnn_col_pi = ['XLC_PI', 'XLE_PI', 'XLI_PI', 'XLS_PI', 'XLV_PI', 'XRC_PI', 'XRE_PI', 'XRI_PI', 'XRS_PI', 'XRV_PI']
cnn_col_ps = ['XLC_PS', 'XLE_PS', 'XLI_PS', 'XLS_PS', 'XLV_PS', 'XRC_PS', 'XRE_PS', 'XRI_PS', 'XRS_PS', 'XRV_PS']
cnn_col_ri = ['XLC_RI', 'XLE_RI', 'XLI_RI', 'XLS_RI', 'XLV_RI', 'XRC_RI', 'XRE_RI', 'XRI_RI', 'XRS_RI', 'XRV_RI']
cnn_col_tav = ['XLC_TAV', 'XLE_TAV', 'XLI_TAV', 'XLS_TAV', 'XLV_TAV', 'XRC_TAV', 'XRE_TAV', 'XRI_TAV', 'XRS_TAV', 'XRV_TAV']