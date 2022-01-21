"""
    purpose: 数据预处理、特征工程
    author: lzz
    date: 20211224
    version: v1
"""

"""
sklearn.decomposition.PCA(n_components=None, copy=True, whiten=False)
参数：
n_components:
    意义：PCA算法中所要保留的主成分个数n，也即保留下来的特征个数n
    类型：int 或者 string，缺省时默认为None，所有成分被保留。
          赋值为int，比如n_components=1，将把原始数据降到一个维度。
          赋值为string，比如n_components='mle'，将自动选取特征个数n，使得满足所要求的方差百分比。
copy:
    类型：bool，True或者False，缺省时默认为True。
    意义：表示是否在运行算法时，将原始训练数据复制一份。若为True，则运行PCA算法后，原始训练数据的值不会有任何改变，因为是在原始数据的副本上进行运算；若为False，则运行PCA算法后，原始训练数据的值会改，因为是在原始数据上进行降维计算。
whiten:
    类型：bool，缺省时默认为False
    意义：白化，使得每个特征具有相同的方差。
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf


class DimReduce:
    """
        数据降维
    """
    def __init__(self, data, corr_thr=0.8, variance_thr=0):
        self.data = data    # 要降维的数据，dataframe格式
        self.corr_thr = corr_thr        # 相关系数阈值
        self.variance_thr = variance_thr        # 方差阈值，当某列方差低于此阈值则删除该列

    def my_pca(self):
        pca = PCA(n_components=3)
        pca.fit(self.data)
        new_data = pca.transform(self.data)
        # new_data = pca.fit_transform(self.data)      # 用data训练模型，同时返回降维后的数据。等价于上面两行
        print(pca.components_)  # 具有最大方差的成分
        print(pca.explained_variance_ratio_)  # 保留成分各自方差的百分比
        """
        output:
        [[-0.13563578 -0.01535013  0.69645167  0.         -0.70450151]
         [ 0.64439682  0.7556768   0.11362091  0.         -0.02820652]
         [ 0.74797434 -0.65471474  0.10614986 -0.         -0.02480319]]
        [0.36726589 0.25403791 0.24457397]
        """
        # 保存模型
        joblib.dump(pca, '/saved_models/pca.pkl')
        return new_data

    def na_count(self):
        """ 各指标缺失规模及缺失占比统计 """
        data_count = self.data.count()   # 每列 非 缺失值(NA)的行数
        na_count = len(self.data) - data_count
        na_rate = na_count / len(self.data)
        result = pd.concat([data_count, na_count, na_rate], axis=1)
        return result

    def miss_data_handle(self):
        """高缺失字段处理"""
        table_col = self.data.columns
        table_col_list = table_col.values.tolist()
        row_length = len(self.data)
        for col_key in table_col_list:
            non_sum1 = self.data[col_key].isnull().sum()     # 多少空行
            if non_sum1 / row_length >= 0.8:
                self.data.drop(col_key, axis=1, inplace=True)    # 删除高缺失值的字段
        return self.data

    def low_variance_filter(self):
        """低方差滤波"""
        var = self.data.var(axis=0)     # 按列求方差
        for i in range(0, len(var)):
            if var[i] <= self.variance_thr:
                self.data.drop(var.index[i], axis=1, inplace=True)
                print("删除低方差列： ", var.index[i])
        return self.data

    def data_corr_analysis(self):
        """相关性分析：返回出原始数据的相关性矩阵以及根据阈值筛选之后的相关性较高的变量"""
        corr_data = self.data.corr()
        for i in range(len(corr_data)):
            for j in range(len(corr_data)):
                if j == i:
                    corr_data.iloc[i, j] = 0

        x, y, corr_xishu = [], [], []
        for i in list(corr_data.index):
            for j in list(corr_data.columns):
                if abs(corr_data.loc[i, j]) > self.corr_thr:
                    x.append(i)
                    y.append(j)
                    corr_xishu.append(corr_data.loc[i, j])
        z = [[x[i], y[i], corr_xishu[i]] for i in range(len(x))]
        high_corr = pd.DataFrame(z, columns=['VAR1', 'VAR2', 'CORR_XISHU'])

        return high_corr


def data_std(data, power_flag=False):
    """
    数据标准化处理
    power_flag: 是否处理功率（大于装机容量的功率以装机容量替代、小于零的以零替代）
    """
    col = list(data.columns)
    data[col] = data[col].apply(pd.to_numeric, errors='coerce').fillna(0.0)  # 把所有列的类型都转化为数值型，出错的地方填入NaN，再把NaN的地方补0，否则会报错ValueError: could not convert string to float
    data = pd.DataFrame(data, dtype='float')

    if power_flag:
        # 大于装机容量的功率应以装机容量替代
        data.grGridActivePower[data['grGridActivePower'] > 3000] = 3000
        # 小于零的功率应以零替代
        data.grGridActivePower[data['grGridActivePower'] < 0] = 0

    # Y = data[['grGridActivePower']]
    # X = data[['grWindSpeed', 'grWindDirction', 'grOutdoorTemperature', 'grAirPressure', 'grAirDensity']]

    # # Z-score标准化
    # scalar = StandardScaler()
    # std_data = scalar.fit_transform(X)

    # # 极差标准化
    # scalar = MinMaxScaler()
    # std_data = scalar.fit_transform(data)
    # data = pd.DataFrame(data=std_data, columns=data.columns, index=data.index)
    # # 保存模型，为了还原
    # joblib.dump(scalar, './saved_models/MinMaxScalar.pkl')

    # 分别对每个字段进行归一化
    for c in col:
        scalar = MinMaxScaler()
        data[c] = scalar.fit_transform(data[c].values.reshape(-1, 1))
        if c == 'grGridActivePower':
            joblib.dump(scalar, './saved_models/MinMaxScalar_y.pkl')    # 保存标签的归一化模型
    print("归一化后的数据示例：\n", data.head())
    return data


def create_dataset(X, y, seq_len=10):
    """seq_len为时间窗口，每seq_len行创建一个block"""
    features = []
    targets = []
    time_index = []

    for i in range(0, len(X) - seq_len, 1):
        data = X.iloc[i:i+seq_len]  # 序列数据
        label = y.iloc[i+seq_len]   # 标签数据  注意不能写错
        time = y.index[i+seq_len]   # 标签时间
        # 保存到features 和 targets
        features.append(data)
        targets.append(label)
        time_index.append(time)

    return np.array(features), np.array(targets), np.array(time_index)    # 要返回array类型


def create_batch_dataset(X, y, train=True, buffer_size=1000, batch_size=32):
    """

    :param X:
    :param y:
    :param train:对于训练集和测试集是不同的操作
    :param buffer_size: 构建批数据时，是否挑选一些窗口数据打乱其数据
    :param batch_size: 构建批数据时，多少窗口数据构建为一批
    :return:
    """
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))   # 数据封装，tensor类型
    if train:   # 训练集
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)  # 训练数据较多，先加载到内存以加快读取速度（比如训练30轮，第1轮从磁盘读取，后面都是从内存读取）
    else:   # 测试集
        return batch_data.batch(batch_size)


def main(data):
    # step1: 去掉无意义列
    col = list(data.columns)
    print("所有字段： ", col)
    for c in col:
        if 'Unnamed' in c:
            data.drop(columns=c, inplace=True)
            print("删除无意义列： ", c)
    # 删除多余的列 hour, year, month
    cand_drop_cols = ['hour', 'year', 'month', 'windspeed_level', 'giWindTurbineOperationMode', 'gbTurbinePowerLimited', 'giWindTurbineYawMode']    # 候选删除列
    drop_cols = []      # 最终删除列
    for col in cand_drop_cols:
        if col in data.columns:
            drop_cols.append(col)
    data.drop(columns=drop_cols, axis=1, inplace=True)
    print("删除多余的列 ", drop_cols)

    # step2: 数据标准化
    data = data_std(data, power_flag=True)

    # step3: 特征工程
    # 特征数据集
    X = data.drop(columns=['grGridActivePower'], axis=1)
    # 标签数据集
    y = data['grGridActivePower']

    # 降维
    dr = DimReduce(data=X)
    X_dr = dr.low_variance_filter()
    print("完成降维，入模特征为： ", list(X_dr.columns))

    # 数据集分离
    X_train, X_test, y_train, y_test = train_test_split(X_dr, y, test_size=0.3, shuffle=False, random_state=666)   # shuffle一定要设成False，不能打乱
    print("训练集X、y，测试集X、y的shape：")
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    # 构造特征数据集（为了满足LSTM数据格式要求）
    seq_len = 16    # 滑窗大小，即每个滑窗有几条数据
    # # 构造训练特征数据集
    train_dataset, train_labels, train_times = create_dataset(X_train, y_train, seq_len=seq_len)
    print("特征数据集，训练集、训练标签、测试集、测试标签的shape：")
    print(train_dataset.shape)  # 分别代表：有多少个滑窗、滑窗大小、每条数据有几个特征
    print(train_labels.shape)
    # # 构造测试特征数据集
    test_dataset, test_labels, test_times = create_dataset(X_test, y_test, seq_len=seq_len)
    print(test_dataset.shape)
    print(test_labels.shape)

    # 构造批数据（为了加快训练速度）
    # # 训练批数据
    train_batch_dataset = create_batch_dataset(train_dataset, train_labels)
    # # 测试批数据
    test_batch_dataset = create_batch_dataset(test_dataset, test_labels, train=False)
    # # 从测试批数据中，获取一个batch_size的样本数据
    # print("测试批数据，其中一个batch_size的样本数据示例：")
    # print(list(test_batch_dataset.as_numpy_iterator())[0])

    return train_dataset, test_dataset, train_labels, test_labels, train_batch_dataset, test_batch_dataset, test_times


if __name__ == '__main__':
    data = pd.read_csv('./data/tmp.csv', dtype={'grGridActivePower': str, 'grWindSpeed': str, 'grWindDirction': str,
                                                'grOutdoorTemperature': str, 'grAirPressure': str, 'grAirDensity': str})
    data.set_index(keys='real_time', inplace=True)
    main(data)

