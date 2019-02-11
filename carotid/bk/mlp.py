from my_util import data_util, plot_util
from carotid import carotid_data_util as cdu
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras import optimizers
from keras import losses
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical


def ann(x_data, y_data):
    nb_feartures = x_data.shape[1]
    nb_classes = 2
    batch_size = int(round(x_data.shape[0]*0.1, 0))
    nb_neuron_1 = int(round(nb_feartures*2/3))
    nb_neuron_2 = int(round(nb_neuron_1*2/3))
    nb_neuron_3 = int(round(nb_neuron_2*2/3))

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
    callbacks_list = [early_stop]

    model = Sequential()
    model.add(Dense(nb_neuron_1, input_dim=nb_feartures))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_neuron_2))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(nb_neuron_3))
    # model.add(Activation('tanh'))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(optimizer=optimizers.sgd(lr=1e-2), loss=losses.mse, metrics=['accuracy'])

    history = model.fit(x_data, to_categorical(y_data),
                        batch_size=batch_size,
                        epochs=500,
                        shuffle=True,
                        validation_split=0.33,
                        callbacks=callbacks_list)
    return model, history


seed = 7
target = 'RCCA'
soure = 'ex'
if(soure == 'exin'):
    id_all, x_data_all, y_data_all = cdu.get_exin_data(target)
    fName = 'ann_exin_'+target
else:
    id_all, x_data_all, y_data_all = cdu.get_ex_data(target)
    fName = 'ann_ex_'+target

# hold out
# id_train, id_test, x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(id_all, x_data_all, y_data_all, test_size=0.3, random_state=seed)
# x_data_train = data_util.scale(x_data_train)
# x_data_test = data_util.scale(x_data_test)
#
# model, history= ann(x_data_train, y_data_train)
# plot_util.plot_acc_loss(history, 'acc')
#
# loss, acc = model.evaluate(x_data_test, to_categorical(y_data_test))
# y_pred = model.predict(x_data_test)
# predict_result_hold = id_test
# predict_result_hold['label'] = y_data_test
# predict_result_hold['0'] = y_pred[:, 0]
# predict_result_hold['1'] = y_pred[:, 1]
# predict_result_hold.to_csv(cdu.get_save_path(fName), sep=',', encoding='utf-8')
# print(acc, loss)


# 10-fold
# kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
# for index, (train, test) in enumerate(kfold.split(x_data_all, y_data_all)):
#     x_train = data_util.scale(x_data_all.iloc[train])
#     x_test = data_util.scale(x_data_all.iloc[test])
#     y_train = y_data_all.iloc[train]
#     model, history= ann(x_train, y_train)
#     loss, acc = model.evaluate(x_test, to_categorical(y_data_all.iloc[test]))
#     y_pred = model.predict(x_test)
#     predict_result_hold = id_all.iloc[test]
#     predict_result_hold['label'] = y_data_all.iloc[test]
#     predict_result_hold['0'] = y_pred[:, 0]
#     predict_result_hold['1'] = y_pred[:, 1]
#     predict_result_hold.to_csv(cdu.get_save_path(fName+'_'+str(index)+'.csv'), sep=',', encoding='utf-8')
#     print(acc, loss)


# leave-one-out
lst = []
scaled_data = data_util.scale(x_data_all)
x_data_all = pd.DataFrame(scaled_data, index=x_data_all.index, columns=x_data_all.columns)
for train, test in LeaveOneOut().split(x_data_all):
    y_train = y_data_all.iloc[train]
    model, history = ann(x_data_all.iloc[train], y_train)
    loss, acc = model.evaluate(x_data_all.iloc[test], to_categorical(y_data_all.iloc[test], 2))
    y_pred = model.predict(x_data_all.iloc[test])
    one_reslut = y_pred[0]
    lst.append([id_all.iloc[test].values[0][0], y_data_all.iloc[test].values[0][0], one_reslut[0], one_reslut[1]])
predict_result = pd.DataFrame(lst, columns=['id', 'label', '0', '1'])
predict_result.to_csv(cdu.get_save_path(fName+'.csv'), sep=',', encoding='utf-8')

print('done')