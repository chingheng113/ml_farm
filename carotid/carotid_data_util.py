from my_util import data_util
import os
import pandas as pd
import numpy as np
from sklearn.utils import resample


def get_ex_data(target):
    df_target = data_util.load_all('Extracranial'+os.sep+target+'_ext_na_ou.csv')
    df_target[target] = 1
    df_normal = data_util.load_all('Extracranial'+os.sep+'normal_ext_na_ou.csv')
    df_normal[target] = 0
    resample_size = df_target.shape[0]
    df_n_downsampled = resample(df_normal,
                                replace=False,    # sample without replacement
                                n_samples=resample_size)     # to match minority class

    df_all = pd.concat([df_target, df_n_downsampled], axis=0).sample(frac=1)
    id = df_all[['ID']]
    x_data = df_all.iloc[:, 1:111]
    y_data = df_all[[target]]
    return id, x_data, y_data


def get_exin_data(target):
    df_target = data_util.load_all('Extracranial+Intracranial'+os.sep+target+'_int_ext_na_ou.csv')
    df_target[target] = 1
    df_normal = data_util.load_all('Extracranial+Intracranial'+os.sep+'normal_int_ext_na_ou.csv')
    df_normal[target] = 0
    resample_size = df_target.shape[0]
    df_n_downsampled = resample(df_normal,
                                replace=False,    # sample without replacement
                                n_samples=resample_size)     # to match minority class

    df_all = pd.concat([df_target, df_n_downsampled], axis=0).sample(frac=1)
    id = df_all[['ID']]
    x_data = df_all.iloc[:, 1:167]
    y_data = df_all[[target]]
    return id, x_data, y_data


def get_ex_all():
    df = data_util.load_all('Extracranial'+os.sep+'ALL_ext_na_ou.csv')
    # df = df.iloc[1:100, :]
    #
    df_n = df[(df['RCCA'] == 0) & (df['REICA'] == 0) & (df['RIICA'] == 0) & (df['RACA'] == 0) &
              (df['RMCA'] == 0) & (df['RPCA'] == 0) & (df['REVA'] == 0) & (df['RIVA'] == 0) &
              (df['BA'] == 0) & (df['LCCA'] == 0) & (df['LEICA'] == 0) & (df['LIICA'] == 0) &
              (df['LACA'] == 0) & (df['LMCA'] == 0) & (df['LPCA'] == 0) & (df['LEVA'] == 0) &
              (df['LIVA'] == 0)]
    df_s = df.drop(index=df_n.index)
    resample_size = df_s.shape[0]
    df_n_downsampled = resample(df_n,
                            replace=False,    # sample without replacement
                            n_samples=resample_size)     # to match minority class

    df_all = pd.concat([df_s, df_n_downsampled], axis=0).sample(frac=1)
    id = df_all[['ID']]
    x_data = df_all.iloc[:, 1:111]
    y_data = df_all.iloc[:, 111:]
    return id, x_data, y_data


def get_ex_in_all():
    df = data_util.load_all('Extracranial+Intracranial'+os.sep+'ALL_int_ext_na_ou.csv')
    #
    df_n = df[(df['RCCA'] == 0) & (df['REICA'] == 0) & (df['RIICA'] == 0) & (df['RACA'] == 0) &
              (df['RMCA'] == 0) & (df['RPCA'] == 0) & (df['REVA'] == 0) & (df['RIVA'] == 0) &
              (df['BA'] == 0) & (df['LCCA'] == 0) & (df['LEICA'] == 0) & (df['LIICA'] == 0) &
              (df['LACA'] == 0) & (df['LMCA'] == 0) & (df['LPCA'] == 0) & (df['LEVA'] == 0) &
              (df['LIVA'] == 0)]
    df_s = df.drop(index=df_n.index)
    resample_size = df_s.shape[0]
    df_n_downsampled = resample(df_n,
                                replace=False,    # sample without replacement
                                n_samples=resample_size)     # to match minority class

    df_all = pd.concat([df_s, df_n_downsampled], axis=0).sample(frac=1)
    id = df_all[['ID']]
    x_data = df_all.iloc[:, 1:166]
    y_data = df_all.iloc[:, 166:]
    return id, x_data, y_data


def get_ex_normal():
    df_normal = data_util.load_all('Extracranial'+os.sep+'normal_ext_na_ou.csv')
    df_normal['normal'] = 0
    id = df_normal[['ID']]
    x_data = df_normal.iloc[:, 1:111]
    y_data = df_normal[['normal']]
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
    get_ex_in_all()
    print('done')
