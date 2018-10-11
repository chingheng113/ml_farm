from my_util import data_util, plot_util
from carotid import carotid_data_util as cdu
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.optimizers import sgd, adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical


dataset = 'done'
target = 'RCCA'
seed = 7
if dataset == 'done':
    id_all, x_data_all, y_data_all = cdu.get_done(target)
    fName = 'svm_new_done.csv'
else:
    id_all, x_data_all, y_data_all = cdu.get_new(target)
    fName = 'svm_new.csv'

id_train, id_test, x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(id_all, x_data_all, y_data_all, test_size=0.3, random_state=seed)
x_data_train = data_util.scale(x_data_train)
x_data_test = data_util.scale(x_data_test)

nb_feartures = x_data_all.shape[1]
nb_classes = 2
batch_size = int(round(x_data_train.shape[0]*0.1, 0))

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
callbacks_list = [early_stop]

model = Sequential()
model.add(Dense(50, input_dim=nb_feartures))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(30))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(nb_classes, activation='sigmoid'))
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_data_train, to_categorical(y_data_train),
                    batch_size=batch_size,
                    epochs=150,
                    shuffle=True,
                    validation_split=0.33,
                    callbacks=callbacks_list)
plot_util.plot_acc_loss(history, 'acc')
loss, acc = model.evaluate(x_data_test, to_categorical(y_data_test))
y_pred = model.predict(x_data_test)
predict_result_hold = id_test
predict_result_hold['label'] = y_data_test
predict_result_hold['0'] = y_pred[:, 0]
predict_result_hold['1'] = y_pred[:, 1]
predict_result_hold.to_csv(cdu.get_save_path(fName), sep=',', encoding='utf-8')
print(acc, loss)
# print(y_pred)
