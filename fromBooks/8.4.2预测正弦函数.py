import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

# 加载matplotlib工具包，使用该工具包可以对预测的sin函数曲线进行绘图
import matplotlib as mpl
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat

from matplotlib import pyplot as plt

learn = tf.contrib.learn

HIDDEN_SIZE = 30  # Lstm中隐藏节点的个数
NUM_LAYERS = 2  # LSTM的层数
TIMESTEPS = 10  # 循环神经网络的截断长度
TRAINING_STEPS = 10000  # 训练轮数
BATCH_SIZE = 32  # batch大小

TRAINING_EXAMPLES = 10000  # 训练数据个数
TESTING_EXAMPLES = 1000  # 测试数据个数
SAMPLE_GAP = 0.01  # 采样间隔


# 定义生成正弦数据的函数
def generate_data(seq):
    X = []
    Y = []
    # 序列的第i项和后面的TIMESTEPS-1项合在一起作为输入;第i+TIMESTEPS项作为输出
    # 即用sin函数前面的TIMESTPES个点的信息，预测第i+TIMESTEPS个点的函数值
    # 注意RNN中输入和标签是来自同一总体数据集，只是有前后顺序而已
    for i in range(len(seq) - TIMESTEPS):       #循环添加9990次
        X.append([seq[i:i + TIMESTEPS]])      #添加输入和标签，这里和之前不一样，输入是一个短句（由多个值构成），标签是一个值
        Y.append([seq[i + TIMESTEPS]])        #这里添加项必须要放在list中，保证三维
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def LstmCell():
    lstm_cell = rnn.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
    return lstm_cell


# 定义lstm模型
def lstm_model(X, y):
    # lstm_cell = rnn.BasicLSTMCell(HIDDEN_SIZE,state_is_tuple=True)
    # cell = rnn.MultiRNNCell([lstm_cell] * NUM_LAYERS)              #最好不要再使用这种方法定义多层RNN,因为这样表示每一层其实用的都是同一个lstm_cell对象

    cell = rnn.MultiRNNCell([LstmCell() for _ in range(NUM_LAYERS)]) #只有这样每次都返回一个新的lstm_cell对象，这样才没问题

    # print("X.shape:",X.shape)      #(?, 1, 10),传入的训练集输入train_X的shape为(9990, 1, 10)
    output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)    #进行RNN的前向传播
    # print("output.shape_post:", output.shape)   #(?, 1, 30)

    output = tf.reshape(output, [-1, HIDDEN_SIZE])
    # print("output.shape_after:", output.shape)  #(?, 30)

    # 通过无激活函数的全连接层计算线性回归，并将数据压缩成一维数组结构,第二个参数表示输出层只有一个节点
    # 这里是将最后一层的最后一次LSTM单元得到的状态进行全连接层的线性回归(状态向量长度等于隐藏层节点数)
    predictions = tf.contrib.layers.fully_connected(output, 1, None)

    # 将predictions和labels调整统一的shape
    # print("y.shape_post:",y.shape)
    labels = tf.reshape(y, [-1])
    # print("y.shape_after:", y.shape)

    # print("predictions.shape_post:", predictions.shape)    #(?, 1)
    predictions = tf.reshape(predictions, [-1])
    # print("predictions.shape_after:", predictions.shape)   #(?,)

    loss = tf.losses.mean_squared_error(predictions, labels)
    train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(),
                                               optimizer="Adagrad",
                                               learning_rate=0.1)
    return predictions, loss, train_op


# 进行训练
# 封装之前定义的lstm
regressor = SKCompat(learn.Estimator(model_fn=lstm_model))
# 生成数据
test_start = TRAINING_EXAMPLES * SAMPLE_GAP
test_end = (TRAINING_EXAMPLES + TESTING_EXAMPLES) * SAMPLE_GAP
train_X, train_y = generate_data(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES, dtype=np.float32))) #这里其实每个样本间隔根本没有0.01，而是(test_start-0)
test_X, test_y = generate_data(np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES, dtype=np.float32))) #但是间隔具体多少并没有太大影响，预测是连续值
# 拟合数据
regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)
# print("train_X.shape:",train_X.shape)    #(9990, 1, 10)
# print("train_y.shape:",train_y.shape)    #(9990, 1)

# 计算预测值
print("test_X.shape:",test_X.shape)   #(990, 1, 10)
preds=regressor.predict(test_X)
print("preds.shape:",preds.shape)     #(990,)

predicted = [[pred] for pred in preds]
print("predicted.len:",len(predicted)) #990

# 计算RMSE
rmse = np.sqrt(((predicted - test_y) ** 2).mean(axis=0))
print("均值方差为:%f" % rmse[0])


fig=plt.figure()
plot_predicted, = plt.plot(predicted, label='predicted')
plot_test, = plt.plot(test_y, label='real_sin')
plt.legend([plot_predicted, plot_test], ['predicted', 'real_sin'])
fig.savefig("sin.png")