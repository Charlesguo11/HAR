import seaborn as sns
from numpy import dstack
from pandas import read_csv
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,GlobalAveragePooling1D,BatchNormalization,MaxPool1D,Activation,Conv1D,LSTM
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


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
    trainX, trainy = inertial_signals_load('train', append_before + 'UCI HAR Dataset/')
    testX, testy = inertial_signals_load('test', append_before + 'UCI HAR Dataset/')
    trainy = trainy - 1
    testy = testy - 1
    print(trainX.shape)
    permutation0 = np.random.permutation(trainy.shape[0])
    trainX = trainX[permutation0, :, :]    
    trainy = trainy[permutation0]
    permutation1=np.random.permutation(testX.shape[0])
    testX = testX[permutation1, :, :]    
    testy = testy[permutation1]
    count_train=np.where(trainy ==5)
    count_test=np.where(testy ==5)
    trainX=np.delete(trainX,count_train[0],axis=0)
    trainy=np.delete(trainy,count_train[0],axis=0)
    testX=np.delete(testX,count_test[0],axis=0)
    testy=np.delete(testy,count_test[0],axis=0)
    trainX=trainX[0:4000]
    trainy=trainy[0:4000]
    testX=testX[0:1300]
    testy=testy[0:1300]
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy

#load data
trainX, trainy, testX, testy = load_dataset()
verbose, epochs, batch_size =1, 1, 250
n_timesteps = trainX.shape[1]
n_features = trainX.shape[2]
n_outputs = trainy.shape[1]
n_steps = 4
n_length = 32
trainX = trainX.reshape((trainX.shape[0], 128,9))
testX = testX.reshape((testX.shape[0], 128,9)) 

#model
model = Sequential()
#lstm layer
model.add(LSTM(100, return_sequences=True, activation='relu',input_shape=(128,9)))
#cnn layer
model.add(Conv1D(filters=64,kernel_size=2, activation='relu', strides=2,name='conv1'))
model.add(MaxPool1D(pool_size=4, padding='same'))
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', strides=1,name='conv2'))#filters=192
model.add(GlobalAveragePooling1D(name='Globalpool'))
model.add(BatchNormalization(epsilon=1e-06,name='Normalize'))
model.add(Dense(5))
model.add(Activation('softmax'))
checkpoint = ModelCheckpoint('lstm_cnn_uci5.h5', monitor='val_accuracy', verbose=1,save_best_only=True,mode='max',period=1) 
callbacks_list = [checkpoint]


#compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
plot_model(model,to_file='model.png',show_shapes=True)
#start training
history=model.fit(trainX, trainy, validation_data=(testX, testy), epochs=epochs, batch_size=batch_size, verbose=verbose,callbacks=callbacks_list)
#save best model
model.save('lstm_cnn_uci5.h5')
loss , accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)
print(model.summary())
score=accuracy*100
#show acc&loss
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


#plot confusion matrix
LABELS = ['WALK','UPSTAIRS','DOWNSTAIRS','SIT','STAND']
ypred_test = model.predict(testX)
print(ypred_test)
max_ypred_test = np.argmax(ypred_test, axis=1)
max_ytest = np.argmax(testy, axis=1)
matrix = metrics.confusion_matrix(max_ytest, max_ypred_test,normalize='true')
plt.figure(figsize=(8, 6))
sns.heatmap(matrix, cmap="YlGnBu", xticklabels=LABELS, yticklabels=LABELS, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
clr=metrics.classification_report(max_ytest, max_ypred_test)
print(clr)


