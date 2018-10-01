import  tensorflow as tf
import numpy as np


# one_cell =  tf.nn.rnn_cell.BasicRNNCell(num_units=128)
#
# cell = tf.nn.rnn_cell.MultiRNNCell([one_cell]*3) # 3层RNN
# print(cell.state_size)    #版本问题，不要再用这种的定义方式

def get_a_cell():
   return   tf.nn.rnn_cell.BasicRNNCell(num_units=128)

# 用tf.nn.rnn_cell MultiRNNCell创建3层RNN
cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(3)]) # 3层RNN
print(cell.state_size)

print("-------------")
# 得到的cell实际也是RNNCell的子类
# 它的state_size是(128, 128, 128)
# (128, 128, 128)并不是128x128x128的意思
# 而是表示共有3个隐层状态，每个隐层状态的大小为128

print(cell.state_size) # (128, 128, 128)

# 使用对应的call函数
inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size

h0 = cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态
print(h0)
print(len(h0))             #三层RNN，存储在一个tuple中
print(h0[0].shape)         #每层的状态大小都是(32, 128)

output, h1 = cell.__call__(inputs, h0)    #或者直接cell(inputs, h0)
#
print(h1) # tuple中含有3个32x128的向量  ，表示三层RNN的下一个时刻的状态