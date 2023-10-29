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
    # trainy = to_categorical(trainy)
    # testy = to_categorical(testy)
    train=(trainX, trainy)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy
    

trainX, trainy, testX, testy = load_dataset()
verbose, epochs, batch_size =1, 30, 192 #192
n_timesteps = trainX.shape[1]
n_features = trainX.shape[2]
n_outputs = trainy.shape[1]
n_steps = 4
n_length = 32
trainX = trainX.reshape((trainX.shape[0], 1152))
testX = testX.reshape((testX.shape[0], 1152)) 
print(trainX.shape,testX.shape)

#SVM model
# from sklearn.svm import SVC
# model = SVC()
# model.fit(trainX,trainy)
# history=model.fit(trainX, trainy)
# predictions = model.predict(testX)
# from sklearn.metrics import accuracy_score
# print('SVM',accuracy_score(testy,predictions)) #0.887343


#DT model
from sklearn import tree #导入需要的模块
clf = tree.DecisionTreeClassifier() #实例化模型对象
clf = clf.fit(trainX,trainy) #用训练集数据训练模型
result = clf.score(testX,testy) #对我们训练的模型精度进行打分
print('DT',result)




