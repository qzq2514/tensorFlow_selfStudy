import numpy as np

X=[1,2]   #输入时间序列，长度为2

state=[0,0]  #初始化上一时刻的输出作为当前时刻输入的一部分


W_cell_state=np.array([[0.1,0.2],[0.3,0.4]])   #上一刻输出作为当前时刻输入部分的权重
W_cell_input=np.array([0.5,0.6])                #当前时刻输入权重
b_cell=np.array([0.1,-0.1])                   #偏置

w_output_full=np.array([[1.0],[2.0]])
b_output=0.1


#按照时间序列顺序执行循环神经网络的前向传播过程

for i in range(len(X)):
    before_active=np.dot(state,W_cell_state)+X[i]*W_cell_input+b_cell
    state=np.tanh(before_active)         #激活函数，作为下一时刻的输入的一部分

    cur_output=np.dot(state,w_output_full)+b_output   #得到当前时刻的输出
    print("%s:before_avtive-"%i,before_active)
    print("%s:state-" % i, state)
    print("%s:cur_output-" % i, cur_output)
    print("--------------------------------")

#具体的示意图详见图"8.1循环神经网络前向传播示意图.png"