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





def file_load(filepath):
    df = read_csv(filepath, header=None, delim_whitespace=True)
    return df.values    

def train_test_append(filenames, append_before=''):
    datalist = list()
    for name in filenames:
        data = file_load(append_before + name)
        datalist.append(data)
    datalist = dstack(datalist)
    return datalist

def inertial_signals_load(group, append_before=''):
    filepath = append_before + group + '/Inertial Signals/'
    filenames = list()
    filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
    filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
    filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
    X = train_test_append(filenames, filepath)
    y = file_load(append_before + group + '/y_'+group+'.txt')
    return X, y
    

def load_dataset(append_before=''):
    trainX, trainy = inertial_signals_load('train', append_before + 'UCI/UCI HAR Dataset/')
    testX, testy = inertial_signals_load('test', append_before + 'UCI/UCI HAR Dataset/')
    trainy = trainy - 1
    testy = testy - 1
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    train=(trainX, trainy )
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy
    

trainX, trainy, testX, testy = load_dataset()
verbose, epochs, batch_size =1, 30, 192 #192
n_timesteps = trainX.shape[1]
n_features = trainX.shape[2]
n_outputs = trainy.shape[1]
n_steps = 4
n_length = 32
trainX = trainX.reshape((trainX.shape[0], 128,9))
testX = testX.reshape((testX.shape[0], 128,9)) 


#model
'''
CNN only: 91.75%
LSTM only: 91.89%
CNN-LSTM: 92.89%
LSTM-CNN: 95.28%
'''
model = Sequential()
#lstm layer
model.add(LSTM(200, return_sequences=True, activation='tanh',input_shape=(128,9)))
#cnn layer
model.add(Conv1D(filters=64,kernel_size=5, activation='relu', strides=2))
model.add(MaxPool1D(pool_size=3, padding='same'))#4
model.add(Conv1D(filters=64, kernel_size=5, activation='relu', strides=2,name='conv'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(150, activation='relu'))
model.add(BatchNormalization(epsilon=1e-06))
model.add(Dense(6))
model.add(Activation('softmax'))
checkpoint = ModelCheckpoint('lstm_cnn_model.h5', monitor='val_accuracy', verbose=1,save_best_only=True,mode='max',period=1) 
callbacks_list = [checkpoint]
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# consturct=plot_model(model,to_file='./model.png',show_shapes=True)
history=model.fit(trainX, trainy, validation_data=(testX, testy), epochs=epochs, batch_size=batch_size, verbose=verbose,callbacks=callbacks_list)
model.summary()
model.save('best.h5')
loss , accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)

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
ypred_test = model.predict(testX)
max_ypred_test = np.argmax(ypred_test, axis=1)
# print(max_ypred_test)
# print(type(max_ypred_test))
# print(max_ypred_test.shape)
max_ypred_test = max_ypred_test.tolist()
max_ytest = np.argmax(testy, axis=1)
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





# location=[]
# for i,x in enumerate(max_ypred_test):
#     if x==5:
#         location.append(i)
# next_start=location[0]
# while True:
#             if max_ypred_test[next_start+1]==5:
#                 next_start=next_start+1
#                 end=next_start
#             else:
#                 break
# print(end)
# for i in range(end+1,len(max_ypred_test)):
# i=end+1
# while 1:
#     if max_ypred_test[i]==5:
#         next_start=i
#         print('i',i)
#         print('end',end)
#         while True:
#             if max_ypred_test[next_start+1]==5:
#                 next_start=next_start+1
#                 end=next_start
#                 i=end+1
                
#             else:
#                 print('111111111111')
#                 break
     
# print(end)
# print(location)
# next_start=location[0]
# print(next_start)

# print([i for i,x in enumerate(max_ypred_test) if x==5])
# print(max_ypred_test.index(5))
# max_ytest = np.argmax(testy, axis=1)
# print(max_ytest)
# matrix = metrics.confusion_matrix(max_ytest, max_ypred_test,normalize='true')
# plt.figure(figsize=(8, 6))
# sns.heatmap(matrix, cmap="YlGnBu", xticklabels=LABELS, yticklabels=LABELS, annot=True)
# plt.title('Confusion Matrix')
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')
# plt.show()
# clr=metrics.classification_report(max_ytest, max_ypred_test)
# print(clr)


