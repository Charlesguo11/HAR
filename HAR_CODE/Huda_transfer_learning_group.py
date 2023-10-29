from tensorflow import keras
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, BatchNormalization,  Activation, Dropout
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import pickle
import csv
from pandas import read_csv
from HuDA_loadData import Huda_process_lt, Huda_process_rt,Huda_process_source,Huda_process_target


'计算 MMD'
def compute_kernel(x, y, kernel_mul=2.0, kernel_num=2):
    '''
    多核或单核高斯核矩阵函数,根据输入样本集x和y,计算返回对应的高斯核矩阵
    Params:
     source: (b1,n)的X分布样本数组
     target:(b2,n)的Y分布样本数组
     kernel_mul: 多核MMD,以bandwidth为中心,两边扩展的基数
     kernel_num: 取不同高斯核的数量
    Return:
      sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = 200
    total = K.concatenate([x, y], axis=0)
    total0 = K.expand_dims(total, axis=0)
    total1 = K.expand_dims(total, axis=1)
    L2_distance = K.sum(K.square(total0 - total1), axis=2)
    bandwidth = K.mean(L2_distance) / float(n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [K.exp(-L2_distance / bandwidth_temp)
                  for bandwidth_temp in bandwidth_list]
    kernel = K.sum(kernel_val)
    return kernel

def compute_mmd(x, y):
    '''
    计算源域数据和目标域数据的MMD距离
    Params: 
	    source: 源域数据(n * len(x))
	    target: 目标域数据(m * len(y))
	Return:
		loss: MMD loss
    '''
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    yx_kernel=compute_kernel(y,x)
    return (K.mean(x_kernel) + K.mean(y_kernel) -  K.mean(xy_kernel)-K.mean(yx_kernel))

def custom_loss(rt_features, lt_features):
    '''
    返回源域数据和目标域数据的MMD距离,取权重为0.0001
    '''
    loss_mmd = compute_mmd(rt_features, lt_features)
    return loss_mmd*0.0001


'以batch的形式获取训练集数据'
def data_generator(batch_size):  # x1:source x2:target y1:source_label y2:target_label
    while 1:
        trainX_rt, trainy_rt, _, _ = Huda_process_source.preprocess()
        trainX_lt, _, _, _ = Huda_process_target.preprocess()
        # trainX_rt = trainX_rt[:19000]
        # trainy_rt = trainy_rt[:19000]
        # trainX_lt = trainX_lt[:19000]
        x1 = trainX_rt
        x2 = trainX_lt
        y1 = trainy_rt
        size = len(y1)
        for i in range(int(size / batch_size)):
            in1 = x1[i*batch_size: (i+1)*batch_size]
            in2 = x2[i*batch_size: (i+1)*batch_size]
            train=K.concatenate([in1, in2], axis=0)
            label = y1[i*batch_size: (i+1)*batch_size]
            yield train, label
    return data_generator

'以batch的形式获取验证集数据'
def val_generator(batch_size):  # x1:source x2:target y1:source_label y2:target_label
    _, _, testx_rt, testy_rt = Huda_process_source.preprocess()
    _, _, testx_lt, testy_lt = Huda_process_target.preprocess()
    # testx_rt = testx_rt[:8000]
    # testy_rt = testy_rt[:8000]
    # testx_lt = testx_lt[:8000]
    # testy_lt = testy_lt[:8000]
    x1 = testx_lt
    x2 = testx_rt
    y1 = testy_lt
    y2 = testy_rt
    size = len(y1)
    for i in range(int(size / batch_size)):
        in1 = x1[i*batch_size: (i+1)*batch_size]
        in2 = x2[i*batch_size: (i+1)*batch_size]
        train=K.concatenate([in1, in2], axis=0)
        out1 = y1[i*batch_size: (i+1)*batch_size]
        out2 = y2[i*batch_size: (i+1)*batch_size]
        label = K.concatenate([out1, out2], axis=0)
        yield train, out1
        

'主函数'
def run():
    '''
    主函数分为数据获取、构建模型、编译模型、绘制val和loss变化图、绘制混淆矩阵
    '''
    '获取训练集和测试集数据'
    trainX_rt, trainy_rt, testx_rt, testy_rt = Huda_process_source.preprocess()
    _, _, testx_lt, testy_lt = Huda_process_target.preprocess()
    # trainX_rt = trainX_rt[:19000]
    # trainy_rt = trainy_rt[:19000]
    # testx_rt = testx_rt[:8000]
    # testy_rt = testy_rt[:8000]
    # testx_lt = testx_lt[:8000]
    # testy_lt = testy_lt[:8000]
    data_size = int(len(trainy_rt))
    batch_size = 100

    '构建迁移学习模型'
    base_model = keras.models.load_model('best_source_group.h5')
    inputs = keras.Input(shape=(80, 6))
    new_model = Model(base_model.input,
                    outputs=base_model.get_layer('Global_pool').output)
    source = new_model(inputs[0:batch_size])
    target = new_model(inputs[batch_size:batch_size*2])
    print(target)
    x = Dropout(0.5)(source)
    x = BatchNormalization(epsilon=1e-06)(x)
    x = Dense(5)(x)
    outputs = Activation('softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    loss_mmd = custom_loss(source, target)
    model.add_loss(loss_mmd)
    model.summary()
    
    '自定义callback函数:用于每一轮训练结束时计算验证集的val和loss'
    class evaluate (keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.val_loss = 0
            self.val_accuracy = 0
            with open('val_acc_and_loss.csv','w') as f:
                csv_write = csv.writer(f)
                csv_head = ["val_loss","val_accuracy"]
                csv_write.writerow(csv_head)

        def on_epoch_end(self, epoch, logs=None):
            self.val_loss, self.val_accuracy = self.model.evaluate(
                val_generator(batch_size), verbose=1)
            with open('val_acc_and_loss.csv','a+') as f:
                csv_write = csv.writer(f)
                data_row = [self.val_loss,self.val_accuracy]
                csv_write.writerow(data_row)
            if self.val_accuracy>0.4:
                model.save('best_target.h5')
                print('best_model saved!')
            print('val_loss  ', format(self.val_loss,'.4f'),' ','val_acc  ', format(self.val_accuracy,'.4f'))

    '编译模型'
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    evaluate_loss = evaluate()
    callbacks_list = [evaluate_loss]
    
    '开始训练'
    history = model.fit(data_generator(batch_size), epochs=15, verbose=1,
                        batch_size=100,steps_per_epoch=int(data_size/batch_size),callbacks=callbacks_list)
    with open('trainHistoryDict.txt', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    loss, accuracy = model.evaluate(val_generator(batch_size), verbose=1)
    print(model.summary())
    print('Accuracy = {}'.format(accuracy))
    print('Loss={}'.format(loss))


    #画出训练过程中的val和loss变化图
    df=read_csv('val_acc_and_loss.csv')
    acc = history.history['accuracy']
    val_acc = df['val_accuracy']
    loss = history.history['loss']
    val_loss = df['val_loss']
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

    #画出混淆矩阵
    confus_model=keras.models.load_model('best_target.h5')
    LABELS = ['Walking','Upstairs','Downstairs','Sitting','Standing']
    _, _, testx_rt, testy_rt = Huda_process_source.preprocess()
    _, _, testx_lt, testy_lt = Huda_process_target.preprocess()
    ypred_test = confus_model.predict(testx_lt)
    max_ypred_test = np.argmax(ypred_test, axis=1)
    max_ytest=np.argmax(testy_lt, axis=1)
    matrix = metrics.confusion_matrix(max_ytest, max_ypred_test, normalize='true')
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, cmap="YlGnBu", xticklabels=LABELS,
                yticklabels=LABELS, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    clr = metrics.classification_report(max_ytest, max_ypred_test)
    print(clr)


if __name__ == '__main__':
    run()
    
