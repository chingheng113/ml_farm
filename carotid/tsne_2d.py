from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_util import data_util, plot_util
from carotid import carotid_data_util as cdu


if __name__ == '__main__':
    dataset = 'done'
    target = 'RCCA'
    seed = 7
    if dataset == 'done':
        id_all, x_data_all, y_data_all = cdu.get_done(target)
        fName = 'svm_new_done.csv'
    else:
        id_all, x_data_all, y_data_all = cdu.get_new(target)
        fName = 'svm_new.csv'

    # calculation
    # x_data_train = data_util.scale(x_data_all)
    # t_sne = TSNE(n_components=2, perplexity=50).fit_transform(x_data_train)
    # df_result = pd.DataFrame(t_sne, columns=['x', 'y'])
    # df_result['label'] = y_data_all.values
    # df_result.to_csv(cdu.get_save_path(target+'_tsne.csv'), sep=',', encoding='utf-8')
    df_result = cdu.get_result(target+'_tsne.csv')
    label = y_data_all.values.ravel()
    n_class = np.unique(label).shape[0]
    plt.figure()
    plt.scatter(df_result.ix[:,0], df_result.ix[:,1], c=label, s=0.2, cmap=plt.cm.get_cmap("jet", n_class))
    plt.colorbar(ticks=range(n_class))
    plt.title('t-SNE 2D visualization: '+ target)
    plt.show()