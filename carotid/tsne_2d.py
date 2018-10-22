from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_util import data_util, plot_util
from carotid import carotid_data_util as cdu


if __name__ == '__main__':
    seed = 7
    id_all, x_data_all, y_data_all = cdu.get_ex_all()
    # calculation
    t_sne = TSNE(n_components=2, perplexity=30).fit_transform(x_data_all)
    result = np.concatenate([t_sne, y_data_all], axis=1)
    df_result = pd.DataFrame(result, columns=['x', 'y']+list(y_data_all.columns.values))
    # df_result['label'] = y_data_all.values
    df_result.to_csv(cdu.get_save_path('ALL_tsne.csv'), sep=',', encoding='utf-8')

    # draw
    target = 'RCCA'
    # df_result = cdu.get_result(target+'_tsne.csv')
    label = y_data_all[target].values.ravel()
    n_class = np.unique(label).shape[0]
    plt.figure()
    plt.scatter(df_result.ix[:,0], df_result.ix[:,1], c=label, s=2, cmap=plt.cm.get_cmap("jet", n_class))
    plt.colorbar(ticks=range(n_class))
    plt.title('t-SNE 2D visualization: '+ target)
    plt.show()