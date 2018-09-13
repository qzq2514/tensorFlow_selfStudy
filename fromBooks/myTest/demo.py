import tensorflow as tf
import numpy as np
import random

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ind=np.arange(10)
np.random.shuffle(ind)

print(ind)

print("---------")

v=tf.Variable(tf.constant(0.1, shape=[1,10]),)

with tf.Session() as sess:

    # print(sess.run(v))
    # print(v.shape)
    #

    print("--------------")
    weights = tf.Variable(tf.constant([[1.0,2,3,4],[3,4,5,6]]))
    weights2 = tf.Variable(tf.constant([[1.0, 2, 3, 4], [3, 4, 5, 6]]))

    print("===============")

    # 此函数大致与tf_nn_softmax_cross_entropy_with_logits的计算方式相同,
    # 适用于每个类别相互独立且排斥的情况，一幅图只能属于一类，而不能同时包含一条狗和一只大象
    #
    # 但是在对于labels的处理上有不同之处, labels从shape来说此函数要求shape为[batch_size],
    # labels[i]
    # 是[0, num_classes)的一个索引, type为int32或int64, 即labels限定了是一个一阶tensor,
    # 并且取值范围只能在分类数之内, 表示一个对象只能属于一个类别

    op = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=weights, labels=tf.argmax(weights, 1))
    sess.run(tf.global_variables_initializer())
    print(sess.run(op))
    print("===============")

    print(weights.shape)
    print(sess.run(tf.argmax(weights,1)))

    print(sess.run(tf.reduce_mean(weights,1)))
    print("--------------")
    a=tf.equal(weights,weights2)
    print(sess.run(a))
    print(sess.run(tf.reduce_mean(tf.cast(a,tf.float32))))

    tf.Variable(0.1)


    print("--------------")

    print(type(weights2.get_shape()))


print("------------")
im={"kk":2,"g":3}
print(len(im.keys()))

a=[x for x in im.keys()]
print(a)

print("-------------")
print(".".join(x for x in["kk","fdf"]))


print("-------------")
pp=np.arange(0,20).reshape([1,4,5])      #从数组的形状中删除单维度条目，即把shape中为1的维度去掉
ppp=np.squeeze(pp)
print(ppp)
for ind,var in enumerate(ppp):
    print(ind,var)

print("-------------")