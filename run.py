"""
    purpose: 风功率预测建模
    author: lzz
    date: 20220111
    version: v1
"""
import datetime
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

from sklearn.metrics import r2_score, mean_squared_error

import getdata
import createDataset

# 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# 解决中文乱码
sns.set_style('whitegrid', {'font.sans-serif': ['simhei', 'FangSong']})


class NeuralNetwork:
    def __init__(self, **kwargs):
        """
        :param **kwargs:
        output_dim=4: output dimension of LSTM layer;
        activation_lstm='tanh': activation function for LSTM layers;
        activation_dense='relu': activation function for Dense layer;
        activation_last='sigmoid': activation function for last layer;
        drop_out=0.2: fraction of input units to drop;
        np_epoch=10, the number of epoches to train the model. epoch is one forward pass and one backward pass of all the training examples;
        batch_size=32: number of samples per gradient update. The higher the batch size, the more memory space you'll need;
        loss='mean_square_error': loss function; optimizer='rmsprop'
        """
        self.output_dim = kwargs.get('output_dim', 8)
        self.activation_lstm = kwargs.get('activation_lstm', 'relu')
        self.activation_dense = kwargs.get('activation_dense', 'relu')
        self.activation_last = kwargs.get('activation_last', 'softmax')    # softmax for multiple output
        self.dense_layer = kwargs.get('dense_layer', 2)     # at least 2 layers
        self.lstm_layer = kwargs.get('lstm_layer', 2)
        self.drop_out = kwargs.get('drop_out', 0.2)
        self.nb_epoch = kwargs.get('nb_epoch', 10)
        self.batch_size = kwargs.get('batch_size', 32)
        self.loss = kwargs.get('loss', 'categorical_crossentropy')
        self.optimizer = kwargs.get('optimizer', 'rmsprop')

    def NN_model(self, trainX, trainY, testX, testY):
        """
        :param trainX: training data set
        :param trainY: expect value of training data
        :param testX: test data set
        :param testY: epect value of test data
        :return: model after training
        """
        print("Training model is LSTM network!")
        input_dim = trainX.shape[1]
        output_dim = trainY.shape[1]    # one-hot label
        # print predefined parameters of current model:
        model = Sequential()
        # applying a LSTM layer with x dim output and y dim input. Use dropout parameter to avoid overfitting
        model.add(LSTM(units=self.output_dim,
                       input_dim=input_dim,
                       activation=self.activation_lstm,
                       dropout=self.drop_out,
                       return_sequences=True))
        for i in range(self.lstm_layer-2):
            model.add(LSTM(units=self.output_dim,
                           input_dim=self.output_dim,
                           activation=self.activation_lstm,
                           dropout=self.drop_out,
                           return_sequences=True))
        # argument return_sequences should be false in last lstm layer to avoid input dimension incompatibility with dense layer
        model.add(LSTM(units=self.output_dim,
                       input_dim=self.output_dim,
                       activation=self.activation_lstm,
                       dropout=self.drop_out))
        for i in range(self.dense_layer-1):
            model.add(Dense(units=self.output_dim, activation=self.activation_last))
        model.add(Dense(units=output_dim, input_dim=self.output_dim, activation=self.activation_last))
        # configure the learning process
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        # train the model with fixed number of epoches
        model.fit(x=trainX, y=trainY, epochs=self.nb_epoch, batch_size=self.batch_size, validation_data=(testX, testY))

        # predict
        predict_y = model.predict(testX)
        print("预测结果：", predict_y)

        score = model.evaluate(trainX, trainY, self.batch_size)
        print("Model evaluation: {}".format(score))

        # # store model to json file
        # model_json = model.to_json()
        # with open(model_path, "w") as json_file:
        #     json_file.write(model_json)
        # # store model weights to hdf5 file
        # if model_weight_path:
        #     if os.path.exists(model_weight_path):
        #         os.remove(model_weight_path)
        #     model.save_weights(model_weight_path)   # eg: model_weight.h5
        return model


def data_vis(data, new_data):
    """
    数据可视化
    data: 原始数据
    new_data: 异常值处理后的数据
    """
    if len(data) < 50000:  # 限定数据条数才画图，否则会卡死
        # 查看风功率变化趋势
        plt.figure(figsize=(16, 8))
        plt.subplot(3, 1, 1)
        plt.plot(data['grGridActivePower'])
        plt.title('风功率变化趋势图')
        plt.xlabel('real_time')

        # 查看特征与标签之间的关系
        plt.subplot(3, 2, 3)
        ax = sns.scatterplot(x='grWindSpeed', y='grGridActivePower', data=data)    # 风速与风功率之间的关系
        ax.set_title('风功率曲线')

        plt.subplot(3, 2, 4)
        ax = sns.lineplot(x='grWindDirction', y='grGridActivePower', data=data)    # 风向与风功率之间的关系
        ax.set_title('风向与风功率之间的关系')

        plt.subplot(3, 3, 7)
        ax = sns.lineplot(x='grOutdoorTemperature', y='grGridActivePower', data=data)    # 室温与风功率之间的关系
        ax.set_title('室温与风功率之间的关系')

        plt.subplot(3, 3, 8)
        ax = sns.pointplot(x='grAirDensity', y='grGridActivePower', data=data)    # 空气密度与风功率之间的关系
        ax.set_title('空气密度与风功率之间的关系')

        plt.subplot(3, 3, 9)
        ax = sns.pointplot(x='month', y='grGridActivePower', data=data)    # 月份与风功率之间的关系
        ax.set_title('月份与风功率之间的关系')
        plt.show()

        plt.figure(figsize=(16, 8))
        ax = sns.pointplot(x='windspeed_level', y='grGridActivePower', data=data, hue='month')    # 基于月份统计风速与风功率之间的关系
        ax.set_title('基于月份统计风速与风功率之间的关系')
        plt.show()

        plt.figure(figsize=(16, 8))
        ax = sns.scatterplot(x='grWindSpeed', y='grGridActivePower', data=new_data)  # 风速与风功率之间的关系
        ax.set_title('风功率曲线（剔除限功率）')
        plt.show()


def create_field(data):
    """创建时间字段，用于分析数据"""
    data['hour'] = data.index.hour
    data['year'] = data.index.year
    data['month'] = data.index.month

    # 风速离散化
    bins = [0, 2, 4, 6, 8, 10, 12, 25]
    s = pd.cut(data['grWindSpeed'], bins)
    data['windspeed_level'] = s
    return data


def create_model(train_dataset, train_batch_dataset, test_batch_dataset):
    # 模型搭建
    model = Sequential([
        LSTM(units=256, input_shape=train_dataset.shape[-2:], return_sequences=True),
        # units:神经元个数；
        # input_shape:输入维度，3维的，第1个维度batch_size不用给出，模型会自动推导，只需给出2个维度：每个滑窗有几条数据、特征维度
        # return_sequences:每次训练后状态要往后传，所以给True
        Dropout(0.4),
        LSTM(units=256, return_sequences=True),
        Dropout(0.2),
        LSTM(units=128, return_sequences=True),
        LSTM(units=32),
        Dense(16)        # 1个预测值
    ])

    # 显示模型结构
    utils.plot_model(model, './saved_models/lstm_model_arc.png')

    # 模型编译
    model.compile(optimizer='adam', loss='mse')
    checkpoint_file = './saved_models/best_model.hdf5'     # 保存最好的模型
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_file,
                                          monitor='loss',
                                          mode='min',
                                          save_best_only=True,
                                          save_weights_only=True)

    # 模型训练
    history = model.fit(train_batch_dataset,
                        epochs=2,       # 迭代次数
                        validation_data=test_batch_dataset,
                        callbacks=[checkpoint_callback])

    # 显示训练结果
    plt.figure(figsize=(16, 8))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend(loc='best')
    plt.show()

    return model


def compute_acc(rmse, cap):
    """
    计算预测准确率
    :param rmse:
    :param cap: 风电场装机容量
    :return:
    """
    return 1 - rmse / cap


def model_val(model, test_dataset, test_labels_std, test_times):
    # print(test_dataset.shape)
    test_preds_std = model.predict(test_dataset, verbose=1)
    # test_preds_std = test_preds_std[:, 0]   # 获取列值，为了把2维预测值转成1维（适用于预测1个点），和真实值保持一致
    print("预测值shape： ", test_preds_std.shape)  # 预测值shape
    print("真实值shape： ", test_labels_std.shape)  # 真实值shape
    print("预测值示例（还原前）： ", test_preds_std[:10])
    print("真实值示例（还原前）： ", test_labels_std[:10])

    # 预测结果还原
    import joblib
    scalar_y = joblib.load('./saved_models/MinMaxScalar_y.pkl')
    # test_preds = scalar_y.inverse_transform(test_preds_std.reshape(-1, 1))
    # test_labels = scalar_y.inverse_transform(test_labels_std.reshape(-1, 1))
    test_preds = scalar_y.inverse_transform(test_preds_std)
    test_labels = scalar_y.inverse_transform(test_labels_std)
    print("预测值示例（还原后）： ", test_preds[:10])
    print("真实值示例（还原后）： ", test_labels[:10])

    # 结果保存
    test_preds_df = pd.DataFrame(data=test_preds, index=test_times[:, 0])
    test_labels_df = pd.DataFrame(data=test_labels, index=test_times[:, 0])
    writer = pd.ExcelWriter('超短时预测结果.xlsx', engine='xlsxwriter')
    test_preds_df.to_excel(writer, sheet_name='预测')
    test_labels_df.to_excel(writer, sheet_name='实际')
    writer.save()
    # df = pd.DataFrame(data=np.concatenate((test_times.reshape(-1, 1), test_preds, test_labels), axis=1),
    #                   columns=['time', 'test_preds', 'test_labels'])
    # df.to_excel('pred_results.xlsx')

    # 计算R2
    score = r2_score(test_labels[:, 0], test_preds[:, 0])
    mse = mean_squared_error(test_labels[:, 0], test_preds[:, 0])
    rmse = math.sqrt(mse)
    acc = compute_acc(rmse, cap=3000*33)       # cap为装机容量
    print("r^2值为：", score)
    print("mse值为：", mse)
    print("rmse值为：", rmse)
    print("准确率为：", acc)

    # 绘制 预测与真实值结果
    plt.figure(figsize=(16, 8))
    plt.plot(test_labels[:1000, 0], label='True value')
    plt.plot(test_preds[:1000, 0], label='Pred value')
    plt.title('预测功率实际功率对比')
    plt.legend(loc='best')
    plt.show()

    return score, mse, rmse, acc


def main():
    st = datetime.datetime.now()
    # 加载数据，查看信息
    # data = getdata.main()
    # data.set_index('real_time', inplace=True)
    data = pd.read_csv('./data/30008/30008_2021_h.csv', parse_dates=['real_time'], index_col=['real_time'])
    # data = pd.read_csv('./data/30008/30008_2021_h.csv', parse_dates=['real_time'], index_col=['wtid', 'real_time'])
    et1 = datetime.datetime.now()
    dur = (et1 - st).seconds
    print("\n读数据耗时：{}秒".format(dur))

    # print("\n****************************查看原始数据信息**********************************")
    # print("数据shape：", data.shape)
    # print("\n数据示例：")
    # print(data.head())
    # print("\n数据集信息，每列非空值个数：")
    # print(data.info())          # 数据集信息，可查看每列非空值
    # print("\n数据分布情况：")
    # print(data.describe())      # 数据集描述，数据分布情况

    # # 创建时间字段
    # print("\n****************************数据增加时间字段、风速离散化**********************************")
    # data = create_field(data)
    # print(data.head().append(data.tail()))

    # 数据可视化
    new_data = createDataset.pro_abnormal(data)   # 异常值处理
    # data_vis(data, new_data)

    # 数据预处理，特征工程
    print("\n****************************数据预处理、特征工程**********************************")
    # 通过数据可视化发现2月数据缺失，为了防止影响建模效果，只选择3月及之后的数据来建模
    # data = new_data[new_data.index.month >= 2]
    new_data.sort_index(inplace=True)
    train_dataset, test_dataset, train_labels, test_labels, train_batch_dataset, test_batch_dataset, test_times = \
        createDataset.main(new_data)
    et2 = datetime.datetime.now()
    dur = (et2 - et1).seconds
    print("\n预处理及特征工程耗时：{}秒".format(dur))

    # 模型训练
    print("\n****************************模型训练**********************************")
    model = create_model(train_dataset, train_batch_dataset, test_batch_dataset)
    et3 = datetime.datetime.now()
    dur = (et3 - et2).seconds
    print("\n模型训练耗时：{}秒".format(dur))

    # 模型验证
    print("\n****************************模型验证**********************************")
    score, mse, rmse, acc = model_val(model, test_dataset, test_labels, test_times)
    et4 = datetime.datetime.now()
    dur = (et4 - et3).seconds
    print("\n模型验证耗时：{}秒".format(dur))

    et = datetime.datetime.now()
    dur = (et-st).seconds
    print("\n完成！总耗时：{}秒".format(dur))


if __name__ == '__main__':
    main()
