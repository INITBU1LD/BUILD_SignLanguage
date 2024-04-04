#reminder that LSTM is an architecture -
#- which is an artificial RNN architecture
#RNNs are used to store and model temporal sequences
#from prior inputs
import os 
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

#creating tensorboard logs to visualize
log_dir = os.path.join('Logs')
tb_callbac = TensorBoard(log_dir=log_dir)

#define the sequential model using keras API in tensorflow
model = Sequential()
model.add(LSTM(64, return_sequences = True, activation = 'relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64,v))