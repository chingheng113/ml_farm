from sklearn import preprocessing as sp
from sklearn.utils import resample
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
import os


def get_model_path(file_name):
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, '..' + os.sep + 'saved_model' + os.sep)
    return os.path.join(filepath + file_name)


def get_file_path(file_name):
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, '..' + os.sep + 'data_source' + os.sep)
    return os.path.join(filepath + file_name)


def load_all(fn):
    read_file_path = get_file_path(fn)
    df = pd.read_csv(read_file_path, encoding='utf8')
    df = df.sample(frac=1)
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
