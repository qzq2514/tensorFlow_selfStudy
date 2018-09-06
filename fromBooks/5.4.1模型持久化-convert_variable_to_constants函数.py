import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
#有时候不需要保存整个模型的所有信息，只需要保存某个指定的结点，可以使用convert_variables_to_constants函数



#保存指定的结点
# v1=tf.Variable(tf.constant(1.0,shape=[1],name="v1"))
# v2=tf.Variable(tf.constant(2.0,shape=[1],name="v2"))
#
# result=v1+v2
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     graph_def=tf.get_default_graph().as_graph_def()
#
#     #给出的计算节点的名称，不需要在后面添加:0
#     output_graph_def=graph_util.convert_variables_to_constants(sess,graph_def,["add"])
#     #必须.pb后缀
#     with tf.gfile.GFile("Saver/combined_model.pb","wb") as f:  #"wb"写入二进制数据
#         f.write(output_graph_def.SerializeToString())



#
with tf.Session() as sess:
    model_filename="Saver/combined_model.pb"
    #读取指定的之前保存好的模型文件
    with tf.gfile.FastGFile(model_filename,"rb") as f:
        #这里也可以像保存时候一样，用tf.get_default_graph().as_graph_def()
        graph_def=tf.GraphDef()
        graph_def.ParseFromString(f.read())

 #保存的时候是保存结点的信息，不需要添加":0",读取加载的时候读取的是张量的信息，需要加上":0"
    result=tf.import_graph_def(graph_def,return_elements=["add:0"])
    print(sess.run(result))

