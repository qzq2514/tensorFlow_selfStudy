import tensorflow as tf
import numpy as np
import random

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
with tf.Session() as sess:
    weights = tf.Variable(tf.constant([[1.0,2,3,4],[3,4,5,6]]))   #(2, 4)

    print("===============")

    # 此函数大致与tf_nn_softmax_cross_entropy_with_logits的计算方式相同,
    # 适用于每个类别相互独立且排斥的情况，一幅图只能属于一类，而不能同时包含一条狗和一只大象
    #
    # 但是在对于labels的处理上有不同之处, labels从shape来说此函数要求shape为[batch_size],
    # labels[i]是[0, num_classes)的一个索引, type为int32或int64, 即labels限定了是一个一阶tensor,
    # 并且取值范围只能在分类数之内, 表示一个对象只能属于一个类别

    #logits：shape为[batch_size,num_classes],type为float32或float64

    #注意这里不是广播，tf.arg_max(weights, 1)是(2,),weights是(2, 4)不符合广播的规则
    op = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=weights, labels=[2,3])
    sess.run(tf.global_variables_initializer())
    print(sess.run(op))
    print("===============")
    print(op.shape)
    print(weights.shape)
    print(sess.run(tf.arg_max(weights,1)))




print("-------------------")
import tensorflow as tf
sess = tf.InteractiveSession()

def index_to_list(index):
    list = np.zeros([2, 3], dtype=int)
    for j in range(2):
        for i in range(3):
            if i == index[j]:
                list[j][i] = 1
            else:
                list[j][i] = 0
    return list

# index = [1, 2]# c = index_to_list(index)# print(c)

a = tf.placeholder(tf.float32, [2, 3])
b = tf.placeholder(tf.int64, [2])
aa = [[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]]
bb = [2, 1]

#sparse_softmax_cross_entropy_with_logits会把labels展开成0,1矩阵，与logits的shape一致
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=a, labels=b))
c = sess.run(cross_entropy, feed_dict={a: [[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]], b: [2, 1]})


a_softmax = tf.nn.softmax(aa)
print(sess.run(a_softmax))     #每行为一组进行softmax

b_list = index_to_list(bb)
print(b_list)

via=-tf.reduce_sum(b_list * tf.log(a_softmax), axis=1)
print(sess.run(via))


loss = tf.reduce_mean(via)
print(sess.run(loss))
print(c)
sess.close()

