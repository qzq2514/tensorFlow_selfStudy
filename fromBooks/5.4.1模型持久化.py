import tensorflow as tf



#保存模型
# v1=tf.Variable(tf.constant([2,3,4]),name="v1")
# v2=tf.Variable(tf.constant([55,66,77]),name="v2")
#
# result=v1+v2
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     saver=tf.train.Saver()
#     saver.save(sess,"Saver/myFirstSaver")   #将数据保存到已有的文件夹中
#     print(sess.run(result))
#     print(result)
    # .meta文件保存计算图结构，.data文件包含训练变量


#加载模型方法一:需要定义和原模型一样的变量，唯一的区别是加载模型不需要运行全局变量的初始化，而是从保存的模型中直接加载进来
#这里的一样是指name一样，不是说变量名也必须一眼该，例如这里定义成vv1,vv2也是可以的
vv1=tf.Variable(tf.constant([2,3,4]),name="other-v1")
vv2=tf.Variable(tf.constant([55,66,77]),name="v2")

result=vv1+vv2

# 如果加载的变量和保存的变量名称不一样，例如，这里v1在定义时name="other-v1",那么这里下面也要哦做点改变，变成:(vv2也要添加进去，不知道为啥)
saver=tf.train.Saver({"v1":vv1,"v2":vv2})
# saver=tf.train.Saver()

with tf.Session() as sess:

    saver.restore(sess,"Saver/myFirstSaver")   #将数据保存到已有的文件夹中
    print(sess.run(result))

#加载模型方法二:
# saver=tf.train.import_meta_graph("Saver/myFirstSaver.meta")
#
# with tf.Session() as sess:
#     saver.restore(sess,"Saver/myFirstSaver")
#     #因为在保存模型中，我们没有将变量放在任何计算图中，也就自动保存到默认图中，然后使用get_tensor_by_name获得指定的张量
#     #因为加法操作的张量retult名字就是add:0
#     print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))







