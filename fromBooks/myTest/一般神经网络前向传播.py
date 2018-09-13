import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#定义隐藏层只有一层的神经网络，每层的神经元个数为2，3，1
W1=tf.Variable(tf.random_normal([3,2],mean=-1,stddev=2,seed=1))
W2=tf.Variable(tf.random_normal([1,3],mean=-1,stddev=2,seed=1))   #定义第一二层的权重参数矩阵，都是正态分布，平均值为-1（默认0）标准差为2，随机种子为1


x=tf.constant([[0.7],[0.9]])  #定义一个样本输入（1*2）

via=tf.matmul(W1,x)
y=tf.matmul(W2,via)     #via,y分别是中间层输出和最终神经网络的输出


sess=tf.Session()


#之前定义的W2，W1都是一个结构，并未有具体的熟知，调用下面两句话后才有真正的初始化值
sess.run(W1.initializer)
sess.run(W2.initializer)      #这里只有两个变量，如果有多个变量，一一这样调用，未免太麻烦，可以调用sess.run(tf.initialize_all_tables)初始化所有的变量

print(via)     #这样直接打印变量，其实是变量的结构，比如这里输出：Tensor("MatMul:0", shape=(3, 1), dtype=float32)
print(sess.run(y)) #想要真正得到变量的值，仍然需要sess.run
sess.close()
