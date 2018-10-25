from my_util import data_util, plot_util
from carotid import carotid_data_util as cdu
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Conv1D, MaxPool1D, Flatten, BatchNormalization, Input
from keras import optimizers
from keras import losses
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras import layers

def cnn(x_ed, x_fv, x_pi, x_ps, x_ri, x_tav, y):
    nb_classes = 2
    batch_size = int(round(x_ed.shape[0]*0.1, 0))
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
    callbacks_list = [early_stop]
    # cnn ED
    nb_feature_ed = x_ed.shape[1]
    cnn_ed_input = Input(shape=(nb_feature_ed, 1))
    conv1_ed = Conv1D(filters=10, kernel_size=2, strides=1, activation='relu')(cnn_ed_input)
    flate_ed = Flatten()(conv1_ed)
    h1_ed = Dense(int(round(nb_feature_ed*2/3, 0)), activation='relu')(flate_ed)
    cnn_ed = Dense(nb_classes)(h1_ed)
    # cnn FV
    nb_feature_fv= x_fv.shape[1]
    cnn_fv_input = Input(shape=(nb_feature_fv, 1))
    conv1_fv = Conv1D(filters=10, kernel_size=2, strides=1, activation='relu')(cnn_fv_input)
    flate_fv = Flatten()(conv1_fv)
    h1_fv = Dense(int(round(nb_feature_fv*2/3, 0)), activation='relu')(flate_fv)
    cnn_fv = Dense(nb_classes)(h1_fv)
    # cnn PI
    nb_feature_pi = x_pi.shape[1]
    cnn_pi_input = Input(shape=(nb_feature_pi, 1))
    conv1_pi = Conv1D(filters=10, kernel_size=2, strides=1, activation='relu')(cnn_pi_input)
    flate_pi = Flatten()(conv1_pi)
    h1_pi = Dense(int(round(nb_feature_pi*2/3, 0)), activation='relu')(flate_pi)
    cnn_pi = Dense(nb_classes)(h1_pi)
    # cnn PS
    nb_feature_ps = x_ps.shape[1]
    cnn_ps_input = Input(shape=(nb_feature_ps, 1))
    conv1_ps = Conv1D(filters=10, kernel_size=2, strides=1, activation='relu')(cnn_ps_input)
    flate_ps = Flatten()(conv1_ps)
    h1_ps = Dense(int(round(nb_feature_ps*2/3, 0)), activation='relu')(flate_ps)
    cnn_ps = Dense(nb_classes)(h1_ps)
    # cnn ri
    nb_feature_ri = x_ri.shape[1]
    cnn_ri_input = Input(shape=(nb_feature_ri, 1))
    conv1_ri = Conv1D(filters=10, kernel_size=2, strides=1, activation='relu')(cnn_ri_input)
    flate_ri = Flatten()(conv1_ri)
    h1_ri = Dense(int(round(nb_feature_ri*2/3, 0)), activation='relu')(flate_ri)
    cnn_ri = Dense(nb_classes)(h1_ri)
    # cnn TAV
    nb_feature_tav = x_tav.shape[1]
    cnn_tav_input = Input(shape=(nb_feature_tav, 1))
    conv1_tav = Conv1D(filters=10, kernel_size=2, strides=1, activation='relu')(cnn_tav_input)
    flate_tav = Flatten()(conv1_tav)
    h1_tav = Dense(int(round(nb_feature_tav*2/3, 0)), activation='relu')(flate_tav)
    cnn_tav = Dense(nb_classes)(h1_tav)
    # merge
    merge = layers.add([cnn_ed, cnn_fv, cnn_pi, cnn_ps, cnn_ri, cnn_tav])
    output = Activation('softmax')(merge)
    model = Model(inputs=[cnn_ed_input, cnn_fv_input, cnn_pi_input, cnn_ps_input, cnn_ri_input, cnn_tav_input],
                  outputs=output)
    model.compile(loss=losses.mse,
                  optimizer=optimizers.sgd(lr=5e-3),
                  metrics=['accuracy'])

    history = model.fit([x_ed, x_fv, x_pi, x_ps, x_ri, x_tav], to_categorical(y),
                        batch_size=batch_size,
                        epochs=300,
                        shuffle=True,
                        validation_split=0.33,
                        callbacks=callbacks_list,
                        verbose=0)
    return model, history

seed = 7
target = 'RMCA'
soure = 'ex'

# 10-fold
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
if(soure == 'exin'):
    id_all, x_data_all, y_data_all = cdu.get_exin_data(target)
    fName = 'cnn_exin_'+target
else:
    id_all, x_data_all, y_data_all = cdu.get_ex_data(target)
    fName = 'cnn_ex_'+target
x_data_ed = x_data_all[data_util.cnn_col_ed]
x_data_fv = x_data_all[data_util.cnn_col_fv]
x_data_pi = x_data_all[data_util.cnn_col_pi]
x_data_ps = x_data_all[data_util.cnn_col_ps]
x_data_ri = x_data_all[data_util.cnn_col_ri]
x_data_tav = x_data_all[data_util.cnn_col_tav]
for index, (train, test) in enumerate(kfold.split(x_data_all, y_data_all)):
    x_train_ed = data_util.scale(x_data_ed.iloc[train])
    x_train_ed = np.expand_dims(x_train_ed, 2)
    x_train_fv = data_util.scale(x_data_fv.iloc[train])
    x_train_fv = np.expand_dims(x_train_fv, 2)
    x_train_pi = data_util.scale(x_data_pi.iloc[train])
    x_train_pi = np.expand_dims(x_train_pi, 2)
    x_train_ps = data_util.scale(x_data_ps.iloc[train])
    x_train_ps = np.expand_dims(x_train_ps, 2)
    x_train_ri = data_util.scale(x_data_ri.iloc[train])
    x_train_ri = np.expand_dims(x_train_ri, 2)
    x_train_tav = data_util.scale(x_data_tav.iloc[train])
    x_train_tav = np.expand_dims(x_train_tav, 2)

    x_test_ed = data_util.scale(x_data_ed.iloc[test])
    x_test_ed = np.expand_dims(x_test_ed, 2)
    x_test_fv = data_util.scale(x_data_fv.iloc[test])
    x_test_fv = np.expand_dims(x_test_fv, 2)
    x_test_pi = data_util.scale(x_data_pi.iloc[test])
    x_test_pi = np.expand_dims(x_test_pi, 2)
    x_test_ps = data_util.scale(x_data_ps.iloc[test])
    x_test_ps = np.expand_dims(x_test_ps, 2)
    x_test_ri = data_util.scale(x_data_ri.iloc[test])
    x_test_ri = np.expand_dims(x_test_ri, 2)
    x_test_tav = data_util.scale(x_data_tav.iloc[test])
    x_test_tav = np.expand_dims(x_test_tav, 2)

    y_train = y_data_all.iloc[train]
    model, history = cnn(x_train_ed, x_train_fv, x_train_pi, x_train_ps, x_train_ri, x_train_tav, y_train)
    loss, acc = model.evaluate([x_test_ed, x_test_fv, x_test_pi, x_test_ps, x_test_ri, x_test_tav], to_categorical(y_data_all.iloc[test]))
    plot_util.plot_acc_loss(history, 'acc')
    y_pred = model.predict([x_test_ed, x_test_fv, x_test_pi, x_test_ps, x_test_ri, x_test_tav])
    predict_result_hold = id_all.iloc[test]
    predict_result_hold['label'] = y_data_all.iloc[test]
    predict_result_hold['0'] = y_pred[:, 0]
    predict_result_hold['1'] = y_pred[:, 1]
    predict_result_hold.to_csv(cdu.get_save_path(fName+'_'+str(index)+'.csv'), sep=',', encoding='utf-8')
    print(acc, loss)
print('done')