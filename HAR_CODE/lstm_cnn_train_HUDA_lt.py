import seaborn as sns
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,GlobalAveragePooling1D,BatchNormalization,MaxPool1D,Activation,Conv1D,LSTM,Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from wisdm_process import wisdm_process
import numpy as np 
from HuDA_loadData import Huda_process_lt



trainX,trainy,testX,testy=Huda_process_lt.preprocess()
verbose, epochs, batch_size =1, 50, 250
n_timesteps = trainX.shape[1]
n_features = trainX.shape[2]
n_outputs = trainy.shape[1]
n_steps = 4
n_length = 32
print(trainX.shape)
trainX = trainX.reshape((trainX.shape[0], 80,3))
testX = testX.reshape((testX.shape[0], 80,3)) 
print(trainX[0].shape,testX[0].shape)
print(trainX[0][1])
print(trainy[0])

#model
model = Sequential()
#lstm layer
model.add(LSTM(100, return_sequences=True, activation='relu',input_shape=(80,3)))
#cnn layer
model.add(Conv1D(filters=64,kernel_size=2, activation='relu', strides=2))
model.add(MaxPool1D(pool_size=4, padding='same'))
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', strides=1,name='conv'))#filters=192
model.add(GlobalAveragePooling1D())
model.add(BatchNormalization(epsilon=1e-06))
model.add(Dense(5))
model.add(Dropout(0.5))
model.add(Activation('softmax'))
checkpoint = ModelCheckpoint('huda_best_model_forwisdm.h5', monitor='val_accuracy', verbose=1,save_best_only=True,mode='max',period=1) 
callbacks_list = [checkpoint]
model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
consturct=plot_model(model,to_file='./model.png',show_shapes=True)
history=model.fit(trainX, trainy, validation_data=(testX, testy), epochs=epochs, batch_size=batch_size, verbose=verbose,callbacks=callbacks_list)
loss , accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)
print(model.summary()) 
score=accuracy*100
print('Accuracy = {}'.format(score))
print('Loss={}'.format(loss))
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.show()



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


model=keras.models.load_model('huda_best_model_forwisdm.h5')
LABELS = ['Walking','Upstairs','Downstairs','Sitting','Standing']
_,_,testX,testy = wisdm_process.preprocess()
ypred_test = model.predict(testX)
max_ypred_test = np.argmax(ypred_test, axis=1)
max_ytest = np.argmax(testy, axis=1)
matrix = metrics.confusion_matrix(max_ytest, max_ypred_test,normalize='true')
plt.figure(figsize=(6, 4))
sns.heatmap(matrix, cmap="YlGnBu", xticklabels=LABELS, yticklabels=LABELS, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
clr=metrics.classification_report(max_ytest, max_ypred_test)
print(clr)

