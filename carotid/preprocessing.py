from my_util import data_util
from sklearn.preprocessing import Imputer
from carotid import carotid_data_util as cdu
import pandas as pd
import numpy as np

def outliers_iqr(ys):
    '''
    http://colingorrie.github.io/outlier-detection.html
    '''
    ys = ys.apply(pd.to_numeric, errors='coerce')
    quartile_1 = ys.quantile(0.25)
    quartile_3 = ys.quantile(0.75)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return (ys > upper_bound) | (ys < lower_bound)


if __name__ == '__main__':
    df = data_util.load_all('carotid_new.csv')
    id = df[['ID']]
    x_data = df.iloc[:, 1:126]
    y_data = df.iloc[:, 126:]

    x_data_n = x_data[y_data['Stenosis_code'] == 0]
    x_data_s = x_data[y_data['Stenosis_code'] == 1]
    for col in x_data:
        x_data_n.loc[outliers_iqr(x_data_n[col]), col] = np.nan
        x_data_s.loc[outliers_iqr(x_data_s[col]), col] = np.nan
    x_data_n2 = Imputer(missing_values=np.nan, strategy='mean', axis=0).fit_transform(x_data_n)
    x_data_s2 = Imputer(missing_values=np.nan, strategy='mean', axis=0).fit_transform(x_data_s)

    x_data_n = pd.DataFrame(x_data_n2, index=x_data_n.index, columns=x_data.columns)
    x_data_s = pd.DataFrame(x_data_s2, index=x_data_s.index, columns=x_data.columns)
    x_data_f = pd.concat([x_data_n, x_data_s], axis=0)

    df_f = pd.concat([id, x_data_f, y_data], axis=1)
    df_f.to_csv(cdu.get_save_path('carotid_new_done.csv'), sep=',', encoding='utf-8')
