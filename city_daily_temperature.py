import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dropout, Dense
import matplotlib.pyplot as plt
import tensorflow as tf

#Sumber Dataset: https://www.kaggle.com/shaneysze/new-york-city-daily-temperature-18692021
data = pd.read_csv(os.path.join('/content', 'nyc_temp_1869_2021.csv'), parse_dates=['MM/DD/YYYY'], index_col='MM/DD/YYYY')
data = data.drop(columns=['Unnamed: 0', 'YEAR', 'MONTH', 'DAY'])
data = data.drop(data.iloc[0:len(data)-12000].index)
data

data.isnull().sum()

data_by_month = data.resample('M').mean()

time = data_by_month.index.values
tmax = data_by_month['TMAX'].values
tmin = data_by_month['TMIN'].values

plt.figure(figsize=(20,5))
plt.plot(time, tmax, label='TMAX')
plt.plot(time, tmin, label='TMIN')
plt.legend()

data_train = data[list(data)[:]].astype(float)
data_train

data_scaled = MinMaxScaler().fit_transform(data_train)
data_scaled

x_train, y_train = [], []
n_future = 1
n_past = 14

for i in range(n_past, len(data_scaled) - n_future + 1):
  x_train.append(data_scaled[i - n_past : i, 0 : data_train.shape[1]])
  y_train.append(data_scaled[i + n_future - 1 : i + n_future, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

print(x_train.shape)
print(y_train.shape)

train_x, test_x, train_y, test_y = train_test_split(x_train, y_train, test_size=0.2, shuffle=False)
print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

model = tf.keras.models.Sequential([
    LSTM(64, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(train_y.shape[1])
])

model.compile(
    loss=tf.keras.losses.Huber(),
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
    metrics=['mae'])

model.summary()

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('mae') < 0.1 and logs.get('val_mae') < 0.1) and epoch >= 10:
      print('\nPada epoch>=10, MAE dan Val_MAE telah mencapai <10% !')
      self.model.stop_training = True

callback = myCallback()

history = model.fit(
    train_x,
    train_y,
    validation_data=(test_x, test_y),
    callbacks=callback,
    epochs=20,
    verbose=1)

plt.figure(figsize=(10,5))
plt.plot(history.epoch, history.history['mae'], label='MAE (Training)')
plt.plot(history.epoch, history.history['val_mae'], label='MAE (Validation)')
plt.title('Plot MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(history.epoch, history.history['loss'], label='Loss (Training)')
plt.plot(history.epoch, history.history['val_loss'], label='Loss (Validation)')
plt.title('Plot Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()