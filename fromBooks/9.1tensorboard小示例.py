import tensorflow as tf


with tf.variable_scope("input1"):               #variable_scope和name_scope在tensorboard下几乎无差别，都可表示命名空间
    input1=tf.constant([1.0,2.0,3.0],name="input1_qzq")
with tf.name_scope("input2"):
    input2=tf.Variable(tf.random_uniform([3]),name="input2_qzq")
output=tf.add_n([input1,input2],name="add_qzq")

writer=tf.summary.FileWriter("log",tf.get_default_graph())
writer.close()


# 在log文件夹父目录中运行tensorboard --logdir="log"
#同时在浏览器中输入网址:http://localhost:6006,不要输入http://qzq2514mbp.local:6006！！！！
