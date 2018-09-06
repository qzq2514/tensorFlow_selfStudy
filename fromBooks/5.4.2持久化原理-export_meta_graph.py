import tensorflow as tf

v1=tf.Variable(tf.constant(1.0,shape=[1],name="v1"))
v2=tf.Variable(tf.constant(2.0,shape=[1],name="v2"))


result=v1+v2
saver=tf.train.Saver()

#export_meta_graph以json格式导出MetaGraph Protocol Buffer
saver.export_meta_graph("Saver/exportMetaGraphDemo.json",as_text=True)



