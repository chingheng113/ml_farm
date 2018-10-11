from my_util import data_util
import os
import pandas as pd
import numpy as np
from sklearn.utils import resample


def get_done(target):
    df = data_util.load_all('carotid_new_done.csv')
    df_s = df[(df['Stenosis_code'] == 1) & (df[target] == 1)]
    resample_size = df_s.shape[0]
    df_n = df[df['Stenosis_code'] == 0]
    df_n_downsampled = resample(df_n,
                                replace=False,    # sample without replacement
                                n_samples=resample_size,     # to match minority class
                                random_state=7) # reproducible results
    resample_inx = pd.concat([df_s, df_n_downsampled], axis=0).sample(frac=1)
    id = df[['ID']].loc[resample_inx.index]
    x_data = df.iloc[:, 1:126].loc[resample_inx.index]
    y_data = df[[target]].loc[resample_inx.index]
    return id, x_data, y_data


def get_new(target):
    df = data_util.load_all('carotid_new.csv')
    df_s = df[(df['Stenosis_code'] == 1) & (df[target] == 1)]
    resample_size = df_s.shape[0]
    df_n = df[df['Stenosis_code'] == 0]
    df_n_downsampled = resample(df_n,
                                replace=False,    # sample without replacement
                                n_samples=resample_size,     # to match minority class
                                random_state=7) # reproducible results
    resample_inx = pd.concat([df_s, df_n_downsampled], axis=0).sample(frac=1)
    id = df[['ID']].loc[resample_inx.index]
    x_data = df.iloc[:, 1:126].loc[resample_inx.index]
    y_data = df[[target]].loc[resample_inx.index]
    return id, x_data, y_data


def get_ko(target):
    df = data_util.load_all('carotid_ko.csv')
    df_s = df[(df['Stenosis_code'] == 1) & (df[target] == 1)]
    resample_size = df_s.shape[0]
    df_n = df[df['Stenosis_code'] == 0]
    df_n_downsampled = resample(df_n,
                                replace=False,    # sample without replacement
                                n_samples=resample_size,     # to match minority class
                                random_state=7) # reproducible results
    resample_inx = pd.concat([df_s, df_n_downsampled], axis=0).sample(frac=1)
    id = df[['ID']].loc[resample_inx.index]
    x_data = df.iloc[:, 1:63].loc[resample_inx.index]
    y_data = df[[target]].loc[resample_inx.index]
    return id, x_data, y_data


def get_save_path(file_name):
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, os.sep + 'results' + os.sep)
    return dirname + os.sep + 'results' + os.sep + file_name


def get_result(fn):
    read_file_path = get_save_path(fn)
    df = pd.read_csv(read_file_path, encoding='utf8')
    return df


if __name__ == '__main__':
    id, x_data, y_data = get_jim2('RCCA')
    predict_result_hold = y_data
    predict_result_hold.to_csv(get_save_path('inx1.csv'), sep=',', encoding='utf-8')
