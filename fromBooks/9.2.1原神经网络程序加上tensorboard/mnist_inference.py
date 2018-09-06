import tensorflow as tf

Input_node=784
Output_node=10
Layer_node=500

#regularizer属于tf.contrib.layers.l2_regularizer
def get_weight_variable(shape,regularizer):
    weights=tf.get_variable(name="weights",shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    #在生成网络权重的参数时，就要确定是否要正则化，正则化则添加到集合losses中
    if regularizer!=None:
        tf.add_to_collection("losses",regularizer(weights))

    return weights

#输入数据，input_tensor得到最后神经网络的输出
def inference(input_tensor,regularizer):
    with tf.variable_scope("layer1"):
        weights=get_weight_variable([Input_node,Layer_node],regularizer)
        biases=tf.get_variable(name="biases",shape=[1,Layer_node],initializer=tf.constant_initializer(0.0))
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights)+biases)


    with tf.variable_scope("layer2"):
        weights=get_weight_variable([Layer_node,Output_node],regularizer)
        biases=tf.get_variable(name="biases",shape=[1,Output_node],initializer=tf.constant_initializer(0.0))

        #输出层不需要使用激活函数
        output=tf.matmul(layer1,weights)+biases

    return output







