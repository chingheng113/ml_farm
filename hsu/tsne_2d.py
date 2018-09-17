from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_util import data_util, plot_util


if __name__ == '__main__':
    seed = 7
    df = data_util.load_all('carotid_modified_1.csv')
    x_data= df.iloc[:, 1:62]
    # calculation
    # x_data_train = data_util.scale(x_data)
    # t_sne = TSNE(n_components=2, perplexity=40).fit_transform(x_data_train)
    # df_result = pd.DataFrame(t_sne, columns=['x', 'y'])
    # df_result['Extra_code'] = df['Extra_code']
    # df_result['Intra_code'] = df['Intra_code']
    # df_result['Ant_code'] = df['Ant_code']
    # df_result['Post_code'] = df['Post_code']
    # df_result['Stenosis_code'] = df['Stenosis_code']
    # df_result['Extra_Intra'] = df['Extra_Intra']
    # df_result['Ant_Post'] = df['Ant_Post']
    # df_result['Stenosis_total'] = df['Stenosis_total']
    # data_util.save_dataframe_to_csv(df_result, 'tsne_2d')

    df_result = pd.read_csv(data_util.get_file_path('tsne_2d.csv'), encoding='utf8')
    label = df['Stenosis_total']
    n_class = np.unique(label).shape[0]
    plt.figure()
    plt.scatter(df_result.ix[:,0], df_result.ix[:,1], c=label, s=0.2, cmap=plt.cm.get_cmap("jet", n_class))
    plt.colorbar(ticks=range(n_class))
    plt.title('t-SNE 2D visualization: Stenosis_total')
    plt.show()