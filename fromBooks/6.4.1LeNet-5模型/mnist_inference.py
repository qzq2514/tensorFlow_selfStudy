import tensorflow as tf

Input_Node=784
Output_node=10   #0～9十种类别

Image_size=28
Num_channels=1    #灰度图，只有一个深度
Num_labels=10



Conv1_Deep=32
Conv1_size=5     #第一个卷积层深度(即过滤器个数)


Conv2_Deep=64
Conv2_size=5     #第二个卷积层深度(即过滤器个数)


FC_size=521     #全连接层的结点数



#input_tensor输入数据的尺寸是:[Batch_size,Image_size,Image_size,Num_channels]
def inference(input_tensor,train,regularizer):
    with tf.variable_scope("layer1-conv1"):
        conv1_weights=tf.get_variable(name="weights",shape=[Conv1_size,Conv1_size,Num_channels,Conv1_Deep],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases=tf.get_variable(name="biases",shape=[Conv1_Deep],initializer=tf.constant_initializer(0.0))


        conv1=tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding="SAME",)
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    with tf.variable_scope("layer2-pool1"):
        pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")


    with tf.variable_scope("layer3-conv2"):   #
        conv2_weights=tf.get_variable(name="weights",shape=[Conv2_size,Conv2_size,Conv1_Deep,Conv2_Deep],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases=tf.get_variable(name="biases",shape=[Conv2_Deep],initializer=tf.constant_initializer(0.0))

        conv2=tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding="SAME")
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))


    with tf.variable_scope("layer4-pool2"):
        pool2=tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    #将第四层输出的矩阵拉成一个长向量，pool_shape[0]样本数,pool_shape[1]~pool_shape[3]是长宽及深度，乘积就是一个样本拉成长向量的长度
    #get_shape()得到是TensorShape类型，可以使用as_list()变成python中的list类型，但是其实没有as_list转换成list也是可以的
    pool_shape=pool2.get_shape().as_list()
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]

    reshaped=tf.reshape(pool2,[pool_shape[0],nodes])   #一行为一个样本



    with tf.variable_scope("layer5-fc1"):
        fc1_weights=tf.get_variable(name="weights",shape=[nodes,FC_size],initializer=tf.truncated_normal_initializer(stddev=0.1))

    #只有全连接层的权重需要加入正则化
    if regularizer!=None:
        tf.add_to_collection("losses",regularizer(fc1_weights))


    fc1_biases=tf.get_variable(name="biases",shape=[FC_size],initializer=tf.constant_initializer(0.1))


    fc1=tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)


    #使用dropout
    if train:
        fc1=tf.nn.dropout(fc1,0.5)


    with tf.variable_scope("layer6-fc2"):
        fc2_weights=tf.get_variable(name="weights",shape=[FC_size,Num_labels],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))

        if regularizer!=None:
            tf.add_to_collection("losses",regularizer(fc2_weights))

        fc2_biases=tf.get_variable("biases",[Num_labels],initializer=tf.constant_initializer(0.1))

        #最后输出层不需要激活函数
        logit=tf.matmul(fc1,fc2_weights)+fc2_biases

    return logit
















