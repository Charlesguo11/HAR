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
from pandas import read_csv
import random
from scipy.stats import mode 
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from HuDA_loadData import Huda_process_lt,Huda_process_rt,Huda_process_source,Huda_process_target,sit_stand_up_stand,sit_stand_up_stand_test


class Huda_process:
    def getRandomIndex(n, x):
	# 索引范围为[0, n), 随机选x个不重复
        index = random.sample(range(n), x)
        return index

    def convert_to_float(x): 
        try: 
            return np.float64(x) 
        except: 
            return np.nan 
    
    def read_data(filepath):
        df=read_csv(filepath,header=None,names=['acc_rt_x','acc_rt_y','acc_rt_z',
                                    'gyro_rt_x','gyro_rt_y','gyro_rt_z',
                                    'act'])
        return df

    #时间序列分割
    def segments(df, time_steps, step, label_name): 
        N_FEATURES = 6
        segments = [] 
        labels = [] 
        for i in range(0, len(df) - time_steps, step): 
            xs = df['acc_rt_x'].values[i:i+time_steps] 
            ys = df['acc_rt_y'].values[i:i+time_steps] 
            zs = df['acc_rt_z'].values[i:i+time_steps] 
            xg = df['gyro_rt_x'].values[i:i+time_steps] 
            yg = df['gyro_rt_y'].values[i:i+time_steps] 
            zg = df['gyro_rt_z'].values[i:i+time_steps] 
    
            label = mode(df[label_name][i:i+time_steps])[0][0] 
            segments.append([xs, ys, zs, xg, yg, zg]) 
            labels.append(label) 
        reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, N_FEATURES) 
        labels = np.asarray(labels) 
        return reshaped_segments,labels

    def preprocess():
        #加载文件
        df = Huda_process.read_data(filepath='/home/gcl/UCI HAR Dataset/UCI HAR Dataset/source_group.csv')   
        df['acc_rt_x'] = (df['acc_rt_x']-df['acc_rt_x'].min())/(df['acc_rt_x'].max()-df['acc_rt_x'].min()) 
        df['acc_rt_y'] = (df['acc_rt_y']-df['acc_rt_y'].min())/(df['acc_rt_y'].max()-df['acc_rt_y'].min()) 
        df['acc_rt_z'] = (df['acc_rt_z']-df['acc_rt_z'].min())/(df['acc_rt_z'].max()-df['acc_rt_z'].min())  
        df['gyro_rt_x'] = (df['gyro_rt_x']-df['gyro_rt_x'].min())/(df['gyro_rt_x'].max()-df['gyro_rt_x'].min()) 
        df['gyro_rt_y'] = (df['gyro_rt_y']-df['gyro_rt_y'].min())/(df['gyro_rt_y'].max()-df['gyro_rt_y'].min()) 
        df['gyro_rt_z'] = (df['gyro_rt_z']-df['gyro_rt_z'].min())/(df['gyro_rt_z'].max()-df['gyro_rt_z'].min())
        #数据分割
        TIME_PERIOD = 80
        STEP_DISTANCE = 40
        LABEL = 'act' 
        data, label = Huda_process.segments(df, TIME_PERIOD, STEP_DISTANCE, LABEL)
        print('data shape:',data.shape)
        df_train,df_test,y_train,y_test=train_test_split(data,label,test_size=0.3) #训练集 训练集标签 测试集 测试集标签
        time_period, sensors = df_train.shape[1], df_train.shape[2] 


        num_classes = 5
        #将所有数据转换为float32
        x_train = df_train.astype('float32') 
        y_train = y_train.astype('float32')
        x_test = df_test.astype('float32') 
        y_test = y_test.astype('float32')
        
        y_train= to_categorical(y_train, num_classes) 
        y_test= to_categorical(y_test, num_classes) 
        
        print('x_train shape:', df_train.shape) 
        print('y_train shape:', y_train.shape) 
        print('x_test shape:', df_test.shape) 
        print('y_test shape:', y_test.shape) 
        return df_train,y_train,df_test,y_test
    

       


trainX,trainy,testX,testy=sit_stand_up_stand.preprocess()
verbose, epochs, batch_size =1, 100, 200
n_timesteps = trainX.shape[1]
n_features = trainX.shape[2]
n_outputs = trainy.shape[1]
n_steps = 4
n_length = 32
print(trainX.shape)
trainX = trainX.reshape((trainX.shape[0], 80,6))
testX = testX.reshape((testX.shape[0], 80,6)) 
print(trainX[0].shape,testX[0].shape)
print(trainX[0][1])
print(trainy[0])

#model
model = Sequential()
#lstm layer
model.add(LSTM(100, return_sequences=True, activation='relu',input_shape=(80,6)))
#cnn layer
model.add(Conv1D(filters=64,kernel_size=2, activation='relu', strides=2))
model.add(MaxPool1D(pool_size=4, padding='same'))
# model.add(Dropout(0.5))
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', strides=1,name='conv'))#filters=192
model.add(GlobalAveragePooling1D(name='Global_pool'))
# model.add(Dropout(0.5))
model.add(BatchNormalization(epsilon=1e-06))
model.add(Dense(5))
model.add(Activation('softmax'))
checkpoint = ModelCheckpoint('best_client01.h5', monitor='val_accuracy', verbose=1,save_best_only=True,mode='max',period=1) 
callbacks_list = [checkpoint]
# lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01,decay_steps=10000,decay_rate=0.96)
model.compile(
            optimizer='adam',
            # optimizer=keras.optimizers.SGD(learning_rate=lr_schedule),
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

# trainX,trainy,testX,testy=Huda_process_lt.preprocess() #利用左右腿进行测试
# trainX,trainy,testX,testy=Huda_process_rt.preprocess() #利用左右腿进行测试
# trainX,trainy,testX,testy=Huda_process_rs.preprocess() #利用右小腿进行测试
# trainX,trainy,testX,testy=Huda_process_target.preprocess() #利用第十组进行测试

# trainX,trainy,testX,testy=sit_stand_up_stand.preprocess() 
model=keras.models.load_model('best_client01.h5')
# LABELS = ['Walking','Upstairs','Downstairs','Sitting','Standing']
LABELS = ['Walking','Sitting','Sitting_down','Standing_up','Standing']
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

