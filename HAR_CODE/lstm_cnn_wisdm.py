import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import mode
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
from numpy import dstack
from pandas import read_csv
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten,GlobalAveragePooling1D,BatchNormalization,MaxPool1D,Activation,Conv1D,LSTM
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pydot
import graphviz


#sns.set(style='whitegrid', palette='muted', font_scale=1.5)

#rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42

columns = ['user','activity','timestamp', 'x-axis', 'y-axis', 'z-axis']
df = pd.read_csv('UCI/UCI HAR Dataset/WISDM_ar_v1.1_raw.txt', header = None, names = columns)
df = df.dropna()

df.head()
df.info()

N_TIME_STEPS = 90
N_FEATURES = 3
step = 20
segments = []
labels = []

for i in range(0, len(df) - N_TIME_STEPS, step):
    xs = df['x-axis'].values[i: i + N_TIME_STEPS]
    ys = df['y-axis'].values[i: i + N_TIME_STEPS]
    zs = df['z-axis'].values[i: i + N_TIME_STEPS]
    label = np.unique(df['activity'][i: i + N_TIME_STEPS])[0]
    segments.append([xs, ys, zs])
    labels.append(label)

reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)

X_train, X_test, y_train, y_test = train_test_split(
        reshaped_segments, labels, test_size=0.2, random_state=RANDOM_SEED)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

verbose, epochs, batch_size =1, 30, 192 #192
'''
CNN :92.51 max:95.49
LSTM : 96.82 max:97.69
LSTM-CNN +BN : 97.73
LSTM-CNN :97.61
'''
#model
model = Sequential()
#lstm layer
model.add(LSTM(200, return_sequences=True, activation='tanh',input_shape=(90,3)))
#cnn layer
model.add(Conv1D(filters=64,kernel_size=5, activation='relu', strides=2))
model.add(MaxPool1D(pool_size=3, padding='same'))#4
model.add(Conv1D(filters=64, kernel_size=5, activation='relu', strides=2,name='conv'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(150, activation='relu'))
# model.add(BatchNormalization(epsilon=1e-06))
model.add(Dense(6))
model.add(Activation('softmax'))
checkpoint = ModelCheckpoint('lstm_cnn_model.h5', monitor='val_accuracy', verbose=1,save_best_only=True,mode='max',period=1) 
callbacks_list = [checkpoint]
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# consturct=plot_model(model,to_file='./model.png',show_shapes=True)
history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=verbose,callbacks=callbacks_list)
model.summary()
model.save('best.h5')
loss , accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)

score=accuracy*100
print('Accuracy = {}'.format(score))
print('Loss={}'.format(loss))


#plot acc&loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

#plot
LABELS = ['WALK','UPSTAIRS','DOWNSTAIRS','SIT','STAND','LAY']
model=keras.models.load_model('lstm_cnn_model.h5')
np.set_printoptions(threshold=1000000)
ypred_test = model.predict(X_test)
max_ypred_test = np.argmax(ypred_test, axis=1)
# print(max_ypred_test)
# print(type(max_ypred_test))
# print(max_ypred_test.shape)
max_ypred_test = max_ypred_test.tolist()
max_ytest = np.argmax(y_test, axis=1)
# print(max_ytest)
matrix = metrics.confusion_matrix(max_ytest, max_ypred_test,normalize='true')
plt.figure(figsize=(8, 6))
sns.heatmap(matrix, cmap="YlGnBu", xticklabels=LABELS, yticklabels=LABELS, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
clr=metrics.classification_report(max_ytest, max_ypred_test)
print(clr)
