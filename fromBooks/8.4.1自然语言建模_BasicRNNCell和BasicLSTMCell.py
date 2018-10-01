import  tensorflow as tf
import numpy as np


cell=tf.nn.rnn_cell.BasicRNNCell(num_units=128)   #参数num_units是隐藏层节点数大小

print(cell.state_size)    #打印出隐藏层节点数大小,也是每个时刻的状态的大小


inputs=tf.placeholder(np.float32,shape=(32,100))

h0=cell.zero_state(32,np.float32)   #通过zero_state得到一个全0的初始状态，形状为(batch_size, state_size)


output, h1 = cell.__call__(inputs, h0) #调用call函数

print(h1.shape)        #输入大小为[32,100],上一状态大小[32,128]所以根据当前输入和上一状态作为本时刻的总体输入，即得到总体输入
                       #大小为[32,228], 隐藏层节点数为128，所以隐藏层参数矩阵大小[228,128]----->隐藏层输出为[32,128]
print(output.shape)



print("----------------")
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)

inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size

h0 = lstm_cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态

# 对于BasicLSTMCell，情况有些许不同，因为LSTM可以看做有两个隐状态h和c，对应的隐层就是一个Tuple，每个都是(batch_size, state_size)的形状：
output, h1 = lstm_cell.__call__(inputs, h0)

print(h1.h)  # shape=(32, 128)
print(h1.c)  # shape=(32, 128)