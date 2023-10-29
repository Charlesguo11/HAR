from pandas import read_csv
import numpy as np 
from scipy.stats import mode 
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import random

'预处理左腿数据'
class Huda_process_lt:
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
        df=read_csv(filepath,header=None,names=['acc_lt_x','acc_lt_y','acc_lt_z',
                                                'gyro_lt_x','gyro_lt_y','gyro_lt_z',
                                                'act'])
        return df

    #时间序列分割
    def segments(df, time_steps, step, label_name): 
        N_FEATURES = 6
        segments = [] 
        labels = [] 
        for i in range(0, len(df) - time_steps, step): 
            xs = df['acc_lt_x'].values[i:i+time_steps] 
            ys = df['acc_lt_y'].values[i:i+time_steps] 
            zs = df['acc_lt_z'].values[i:i+time_steps] 
            xg = df['gyro_lt_x'].values[i:i+time_steps] 
            yg = df['gyro_lt_y'].values[i:i+time_steps] 
            zg = df['gyro_lt_z'].values[i:i+time_steps] 
    
            label = mode(df[label_name][i:i+time_steps])[0][0] 
            segments.append([xs, ys, zs, xg, yg, zg]) 
            labels.append(label) 
        reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, N_FEATURES) 
        labels = np.asarray(labels) 
        return reshaped_segments,labels

    def preprocess():
        #加载文件
        df = Huda_process_lt.read_data(filepath='/home/gcl/UCI HAR Dataset/UCI HAR Dataset/lt_ultimate_transfer.csv')   
        df['acc_lt_x'] = (df['acc_lt_x']-df['acc_lt_x'].min())/(df['acc_lt_x'].max()-df['acc_lt_x'].min()) 
        df['acc_lt_y'] = (df['acc_lt_y']-df['acc_lt_y'].min())/(df['acc_lt_y'].max()-df['acc_lt_y'].min()) 
        df['acc_lt_z'] = (df['acc_lt_z']-df['acc_lt_z'].min())/(df['acc_lt_z'].max()-df['acc_lt_z'].min())  
        df['gyro_lt_x'] = (df['gyro_lt_x']-df['gyro_lt_x'].min())/(df['gyro_lt_x'].max()-df['gyro_lt_x'].min()) 
        df['gyro_lt_y'] = (df['gyro_lt_y']-df['gyro_lt_y'].min())/(df['gyro_lt_y'].max()-df['gyro_lt_y'].min()) 
        df['gyro_lt_z'] = (df['gyro_lt_z']-df['gyro_lt_z'].min())/(df['gyro_lt_z'].max()-df['gyro_lt_z'].min())
        #数据分割
        TIME_PERIOD = 80
        STEP_DISTANCE = 40
        LABEL = 'act' 
        data, label = Huda_process_lt.segments(df, TIME_PERIOD, STEP_DISTANCE, LABEL)
        # print('data shape:',data.shape)
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
        return df_train,y_train,df_test,y_test
    
    
'预处理右腿数据'
class Huda_process_rt:
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
        df = Huda_process_rt.read_data(filepath='/home/gcl/UCI HAR Dataset/UCI HAR Dataset/ultimate.csv')   
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
        data, label = Huda_process_rt.segments(df, TIME_PERIOD, STEP_DISTANCE, LABEL)
        # print('data shape:',data.shape)
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
        return df_train,y_train,df_test,y_test
    


'预处理右小腿数据'
class Huda_process_rs:
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
        df=read_csv(filepath,header=None,names=['acc_rs_x','acc_rs_y','acc_rs_z',
                                                'gyro_rs_x','gyro_rs_y','gyro_rs_z',
                                                'act'])
        return df

    #时间序列分割
    def segments(df, time_steps, step, label_name): 
        N_FEATURES = 6
        segments = [] 
        labels = [] 
        for i in range(0, len(df) - time_steps, step): 
            xs = df['acc_rs_x'].values[i:i+time_steps] 
            ys = df['acc_rs_y'].values[i:i+time_steps] 
            zs = df['acc_rs_z'].values[i:i+time_steps] 
            xg = df['gyro_rs_x'].values[i:i+time_steps] 
            yg = df['gyro_rs_y'].values[i:i+time_steps] 
            zg = df['gyro_rs_z'].values[i:i+time_steps] 
    
            label = mode(df[label_name][i:i+time_steps])[0][0] 
            segments.append([xs, ys, zs, xg, yg, zg]) 
            labels.append(label) 
        reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, N_FEATURES) 
        labels = np.asarray(labels) 
        return reshaped_segments,labels

    def preprocess():
        #加载文件
        df = Huda_process_rs.read_data(filepath='/home/gcl/UCI HAR Dataset/UCI HAR Dataset/rs_ultimate_transfer.csv')   
        df['acc_rs_x'] = (df['acc_rs_x']-df['acc_rs_x'].min())/(df['acc_rs_x'].max()-df['acc_rs_x'].min()) 
        df['acc_rs_y'] = (df['acc_rs_y']-df['acc_rs_y'].min())/(df['acc_rs_y'].max()-df['acc_rs_y'].min()) 
        df['acc_rs_z'] = (df['acc_rs_z']-df['acc_rs_z'].min())/(df['acc_rs_z'].max()-df['acc_rs_z'].min())  
        df['gyro_rs_x'] = (df['gyro_rs_x']-df['gyro_rs_x'].min())/(df['gyro_rs_x'].max()-df['gyro_rs_x'].min()) 
        df['gyro_rs_y'] = (df['gyro_rs_y']-df['gyro_rs_y'].min())/(df['gyro_rs_y'].max()-df['gyro_rs_y'].min()) 
        df['gyro_rs_z'] = (df['gyro_rs_z']-df['gyro_rs_z'].min())/(df['gyro_rs_z'].max()-df['gyro_rs_z'].min())
        #数据分割
        TIME_PERIOD = 80
        STEP_DISTANCE = 40
        LABEL = 'act' 
        data, label = Huda_process_rs.segments(df, TIME_PERIOD, STEP_DISTANCE, LABEL)
        # print('data shape:',data.shape)
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
    

'预处理1-9号用户数据'
class Huda_process_source:
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
        df=read_csv(filepath,header=None,names=['acc_rs_x','acc_rs_y','acc_rs_z',
                                                'gyro_rs_x','gyro_rs_y','gyro_rs_z',
                                                'act'])
        return df

    #时间序列分割
    def segments(df, time_steps, step, label_name): 
        N_FEATURES = 6
        segments = [] 
        labels = [] 
        for i in range(0, len(df) - time_steps, step): 
            xs = df['acc_rs_x'].values[i:i+time_steps] 
            ys = df['acc_rs_y'].values[i:i+time_steps] 
            zs = df['acc_rs_z'].values[i:i+time_steps] 
            xg = df['gyro_rs_x'].values[i:i+time_steps] 
            yg = df['gyro_rs_y'].values[i:i+time_steps] 
            zg = df['gyro_rs_z'].values[i:i+time_steps] 
    
            label = mode(df[label_name][i:i+time_steps])[0][0] 
            segments.append([xs, ys, zs, xg, yg, zg]) 
            labels.append(label) 
        reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, N_FEATURES) 
        labels = np.asarray(labels) 
        return reshaped_segments,labels

    def preprocess():
        #加载文件
        df = Huda_process_rs.read_data(filepath='/home/gcl/UCI HAR Dataset/UCI HAR Dataset/source_group.csv')   
        df['acc_rs_x'] = (df['acc_rs_x']-df['acc_rs_x'].min())/(df['acc_rs_x'].max()-df['acc_rs_x'].min()) 
        df['acc_rs_y'] = (df['acc_rs_y']-df['acc_rs_y'].min())/(df['acc_rs_y'].max()-df['acc_rs_y'].min()) 
        df['acc_rs_z'] = (df['acc_rs_z']-df['acc_rs_z'].min())/(df['acc_rs_z'].max()-df['acc_rs_z'].min())  
        df['gyro_rs_x'] = (df['gyro_rs_x']-df['gyro_rs_x'].min())/(df['gyro_rs_x'].max()-df['gyro_rs_x'].min()) 
        df['gyro_rs_y'] = (df['gyro_rs_y']-df['gyro_rs_y'].min())/(df['gyro_rs_y'].max()-df['gyro_rs_y'].min()) 
        df['gyro_rs_z'] = (df['gyro_rs_z']-df['gyro_rs_z'].min())/(df['gyro_rs_z'].max()-df['gyro_rs_z'].min())
        #数据分割
        TIME_PERIOD = 80
        STEP_DISTANCE = 40
        LABEL = 'act' 
        data, label = Huda_process_rs.segments(df, TIME_PERIOD, STEP_DISTANCE, LABEL)
        # print('data shape:',data.shape)
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
    
        return df_train,y_train,df_test,y_test



'预处理10号用户数据'
class Huda_process_target:
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
        df=read_csv(filepath,header=None,names=['acc_rs_x','acc_rs_y','acc_rs_z',
                                                'gyro_rs_x','gyro_rs_y','gyro_rs_z',
                                                'act'])
        return df

    #时间序列分割
    def segments(df, time_steps, step, label_name): 
        N_FEATURES = 6
        segments = [] 
        labels = [] 
        for i in range(0, len(df) - time_steps, step): 
            xs = df['acc_rs_x'].values[i:i+time_steps] 
            ys = df['acc_rs_y'].values[i:i+time_steps] 
            zs = df['acc_rs_z'].values[i:i+time_steps] 
            xg = df['gyro_rs_x'].values[i:i+time_steps] 
            yg = df['gyro_rs_y'].values[i:i+time_steps] 
            zg = df['gyro_rs_z'].values[i:i+time_steps] 
    
            label = mode(df[label_name][i:i+time_steps])[0][0] 
            segments.append([xs, ys, zs, xg, yg, zg]) 
            labels.append(label) 
        reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, N_FEATURES) 
        labels = np.asarray(labels) 
        return reshaped_segments,labels

    def preprocess():
        #加载文件
        df = Huda_process_rs.read_data(filepath='/home/gcl/UCI HAR Dataset/UCI HAR Dataset/target_group.csv')   
        df['acc_rs_x'] = (df['acc_rs_x']-df['acc_rs_x'].min())/(df['acc_rs_x'].max()-df['acc_rs_x'].min()) 
        df['acc_rs_y'] = (df['acc_rs_y']-df['acc_rs_y'].min())/(df['acc_rs_y'].max()-df['acc_rs_y'].min()) 
        df['acc_rs_z'] = (df['acc_rs_z']-df['acc_rs_z'].min())/(df['acc_rs_z'].max()-df['acc_rs_z'].min())  
        df['gyro_rs_x'] = (df['gyro_rs_x']-df['gyro_rs_x'].min())/(df['gyro_rs_x'].max()-df['gyro_rs_x'].min()) 
        df['gyro_rs_y'] = (df['gyro_rs_y']-df['gyro_rs_y'].min())/(df['gyro_rs_y'].max()-df['gyro_rs_y'].min()) 
        df['gyro_rs_z'] = (df['gyro_rs_z']-df['gyro_rs_z'].min())/(df['gyro_rs_z'].max()-df['gyro_rs_z'].min())
        #数据分割
        TIME_PERIOD = 80
        STEP_DISTANCE = 40
        LABEL = 'act' 
        data, label = Huda_process_rs.segments(df, TIME_PERIOD, STEP_DISTANCE, LABEL)
        # print('data shape:',data.shape)
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
        
        return df_train,y_train,df_test,y_test
    
    
'预处理01号用户的四种动作训练数据'
class sit_stand_up_stand:
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
        df = sit_stand_up_stand.read_data(filepath='/home/gcl/UCI HAR Dataset/UCI HAR Dataset/client01_train_last.csv')   
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
        data, label = sit_stand_up_stand.segments(df, TIME_PERIOD, STEP_DISTANCE, LABEL)
        # print('data shape:',data.shape)
        df_train,df_test,y_train,y_test=train_test_split(data,label,test_size=0.2) #训练集 训练集标签 测试集 测试集标签
        time_period, sensort = df_train.shape[1], df_train.shape[2] 


        num_classes = 5
        #将所有数据转换为float32
        x_train = df_train.astype('float32') 
        y_train = y_train.astype('float32')
        x_test = df_test.astype('float32') 
        y_test = y_test.astype('float32')
        
        y_train= to_categorical(y_train, num_classes) 
        y_test= to_categorical(y_test, num_classes) 
        
        return df_train,y_train,df_test,y_test
    
'预处理01号用户的四种动作测试数据'
class sit_stand_up_stand_test:
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
        df = Huda_process_rt.read_data(filepath='/home/gcl/UCI HAR Dataset/UCI HAR Dataset/client01_test_last.csv')   
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
        data, label = Huda_process_rt.segments(df, TIME_PERIOD, STEP_DISTANCE, LABEL)
        # print('data shape:',data.shape)
        df_train,df_test,y_train,y_test=train_test_split(data,label,test_size=0.3) #训练集 训练集标签 测试集 测试集标签
        time_period, sensort = df_train.shape[1], df_train.shape[2] 


        num_classes = 4
        #将所有数据转换为float32
        x_train = df_train.astype('float32') 
        y_train = y_train.astype('float32')
        x_test = df_test.astype('float32') 
        y_test = y_test.astype('float32')
        
        y_train= to_categorical(y_train, num_classes) 
        y_test= to_categorical(y_test, num_classes) 
        
        return df_train,y_train,df_test,y_test