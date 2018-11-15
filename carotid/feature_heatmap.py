import csv, os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


def cross_selected(df_all):
    threshold = 10
    for target in targets:
        a = df_all.loc[(df_all[target] != threshold) | (df_all[target] < threshold)]
        print('-')



# https://zhuanlan.zhihu.com/p/28447106
targets = ['RCCA', 'REICA', 'RIICA', 'RACA', 'RMCA', 'RPCA', 'REVA', 'RIVA', 'BA', 'LCCA', 'LEICA', 'LIICA', 'LACA',
           'LMCA', 'LPCA', 'LEVA', 'LIVA']
soure = 'exin'
for index, target in enumerate(targets):
    all_selected_features = []
    with open('fs'+os.sep+target+'_'+soure+'_fs.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            for f in row:
                all_selected_features.append(f)
    feature_dict = Counter(all_selected_features)
    feature_df = pd.DataFrame.from_dict(feature_dict, orient='index', columns=[target])
    if index == 0:
        df_all = feature_df
    else:
        df_all = pd.concat([df_all, feature_df], axis=1)
df_all.replace(np.nan, 0, inplace=True)
# top 20
df_all['selected_total'] = df_all. sum(axis=1)
df_all.sort_values(by=['selected_total'], ascending=False, inplace=True)
df_all = df_all.iloc[0:20,:]
df_all.drop(['selected_total'], inplace=True, axis=1)

f, ax = plt.subplots(figsize = (50, df_all.shape[0]))
cmap = sns.cubehelix_palette(start=1, rot=3, gamma=0.8, as_cmap=True)
g = sns.heatmap(df_all, cmap=cmap, linewidths=0.05, ax=ax)
if soure == 'ex':
    ax.set_title('Feature Selection Heatmap of Extracranial Inputs')
else:
    ax.set_title('Feature Selection Heatmap of Extracranial and Intracranial Inputs')
ax.set_xlabel('Dataset')
ax.set_ylabel('Feature')
ax.set_yticklabels(g.get_yticklabels(), rotation=40, fontsize=10)
plt.show()
print('done')