"""
    purpose: 风功率预测建模
    author: lzz
    date: 20220111
    version: v1
"""
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

from sklearn.metrics import r2_score

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


def data_vis(data):
    """数据可视化"""
    # 查看风功率变化趋势
    if len(data) < 10000:  # 限定数据条数才画图，否则会卡死
        plt.figure(figsize=(16, 8))
        plt.plot(data['grGridActivePower'])
        plt.title('grGridActivePower')
        plt.xlabel('real_time')
        plt.show()

    # 查看特征与标签之间的关系
    if len(data) < 10000:  # 限定数据条数才画图，否则会卡死
        plt.figure(figsize=(16, 8))
        ax = sns.pointplot(x='grWindSpeed', y='grGridActivePower', data=data)    # 风速与风功率之间的关系
        ax.set_title('风速与风功率之间的关系')
        plt.show()

        # plt.figure(figsize=(16, 8))
        # ax = sns.pointplot(x='grWindDirction', y='grGridActivePower', data=data)    # 风向与风功率之间的关系
        # ax.set_title('风向与风功率之间的关系')
        # plt.show()

        plt.figure(figsize=(16, 8))
        ax = sns.lineplot(x='grOutdoorTemperature', y='grGridActivePower', data=data)    # 室温与风功率之间的关系
        ax.set_title('室温与风功率之间的关系')
        plt.show()

        plt.figure(figsize=(16, 8))
        ax = sns.pointplot(x='grAirDensity', y='grGridActivePower', data=data)    # 空气密度与风功率之间的关系
        ax.set_title('空气密度与风功率之间的关系')
        plt.show()

        # plt.figure(figsize=(16, 8))
        # ax = sns.lineplot(x='month', y='grGridActivePower', data=data)    # 月份与风功率之间的关系
        # ax.set_title('月份与风功率之间的关系')
        # plt.show()

        plt.figure(figsize=(16, 8))
        ax = sns.pointplot(x='grWindSpeed', y='grGridActivePower', data=data, hue='month')    # 基于月份统计风速与风功率之间的关系
        ax.set_title('基于月份统计风速与风功率之间的关系')
        plt.show()


def create_time(data):
    """创建时间字段，用于分析数据"""
    data['hour'] = data.index.hour
    data['year'] = data.index.year
    data['month'] = data.index.month
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
        Dropout(0.3),
        LSTM(units=128, return_sequences=True),
        LSTM(units=32),
        Dense(1)        # 1个预测值
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
                        epochs=10,
                        validation_data=test_batch_dataset,
                        callbacks=[checkpoint_callback])

    # 显示训练结果
    plt.figure(figsize=(16, 8))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend(loc='best')
    plt.show()

    return model


def model_val(model, test_dataset, test_labels):
    print(test_dataset.shape)
    test_preds = model.predict(test_dataset, verbose=1)
    print(test_preds.shape)
    print(test_preds[:10])
    test_preds = test_preds[:, 0]   # 获取列值，为了把2维预测值转成1维，和真实值保持一致
    print(test_preds[:10])
    print(test_preds.shape)         # 预测值shape
    print(test_labels.shape)        # 真实值shape

    # 计算R2
    score = r2_score(test_labels, test_preds)
    print("r^2值为：", score)

    # 绘制 预测与真实值结果
    plt.figure(figsize=(16, 8))
    plt.plot(test_labels, label='True value')
    plt.plot(test_preds, label='Pred value')
    plt.legend(loc='best')
    plt.show()

    return score


def main():
    st = datetime.datetime.now()
    # 加载数据，查看信息
    data = pd.read_csv('./data/tmp.csv', parse_dates=['real_time'], index_col=['real_time'])
    # data = getdata.main()
    # data.set_index('real_time', inplace=True)
    print("\n****************************查看原始数据信息**********************************")
    print(data.shape)
    print(data.head())
    print(data.info())          # 数据集信息，可查看每列非空值
    print(data.describe())      # 数据集描述，数据分布情况

    # 创建时间字段
    print("\n****************************数据增加时间字段**********************************")
    data = create_time(data)
    print(data.head().append(data.tail()))

    # 数据可视化
    data_vis(data)

    # 数据预处理，特征工程
    print("\n****************************数据预处理、特征工程**********************************")
    train_dataset, test_dataset, train_labels, test_labels, train_batch_dataset, test_batch_dataset = createDataset.main(data)

    # 模型训练
    print("\n****************************模型训练**********************************")
    model = create_model(train_dataset, train_batch_dataset, test_batch_dataset)

    # 模型验证
    print("\n****************************模型验证**********************************")
    score = model_val(model, test_dataset, test_labels)

    et = datetime.datetime.now()
    dur = (et-st).seconds
    print("\n完成！耗时：{}秒".format(dur))


if __name__ == '__main__':
    main()
