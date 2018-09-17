from my_util import data_util, plot_util
from keras.utils import to_categorical
import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.optimizers import sgd, adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

seed = 7
np.random.seed(seed)
fn = 'kc.csv'
read_file_path = data_util.get_file_path(fn)
df = pd.read_csv(read_file_path, encoding='utf8')
# shuffle samples
df = df.sample(frac=1)
df = df[['KCtimes', 'G4_count', 'Penta_count', 'ORBS_count', 'exam_times', 'miss_exam', 'func_date', 'dob', 'sex', 'group']]
df = df.dropna()
#x_data
x_data = df[['KCtimes', 'G4_count', 'Penta_count', 'ORBS_count', 'exam_times', 'miss_exam']]
b_day = pd.to_datetime(df['dob'], format='%m/%d/%y', errors='coerce')
for i, b in b_day.items():
    if b.year > 2018:
        b_day[i] = b_day[i].replace(year=b.year-100)
onset_day = pd.to_datetime(df['func_date'], format='%m/%d/%y', errors='coerce')
AGE = np.floor((onset_day - b_day) / pd.Timedelta(days=365))
x_data['onset_age'] = AGE
sex_dumy = pd.get_dummies(df['sex'], prefix='sex')
x_data = pd.concat([sex_dumy, x_data], axis=1)
x_data = data_util.scale(x_data)
# y_data
y_data = to_categorical(df['group']-1)

# make train and test
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
# model
nb_features = x_train.shape[1]
nb_classes = y_train.shape[1]
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=1000, verbose=1, mode='auto')
callbacks_list = [early_stop]

model = Sequential()
model.add(Dense(9, input_dim=nb_features))
model.add(BatchNormalization())
model.add(Activation('relu', name='relu_1'))
model.add(Dropout(0.))
model.add(Dense(5, input_dim=nb_features))
model.add(BatchNormalization())
model.add(Activation('relu', name='relu_2'))
model.add(Dropout(0.))
model.add(Dense(units=nb_classes))
model.add(Activation('softmax', name='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=adam(),
              metrics=['categorical_accuracy'])
plot_util.plot_model(model, 'test')
history = model.fit(x_train, y_train,
                    batch_size=500,
                    epochs=2000,
                    shuffle=True,
                    validation_split=0.33,
                    callbacks=callbacks_list)

plot_util.plot_acc_loss(history, 'categorical_accuracy')

# confusion matrix
y_predict = model.predict(x_test)
y_test = np.argmax(y_test, axis=1).astype('str')
y_pred = np.argmax(y_predict, axis=1).astype('str')
C = confusion_matrix(y_test, y_pred)
print(C)
print(classification_report(y_test,y_pred))
# Plot non-normalized confusion matrix
