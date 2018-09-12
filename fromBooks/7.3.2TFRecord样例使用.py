import tensorflow as tf

# 在7.3.1TFRecord样例生成得到TFRecord_Dir/threadLearning中的两个文件后，
# 本程序将文件中保存的数据进行读取


files=tf.train.match_filenames_once("TFRecord_Dir/threadLearning/data.tfrecord-*")  #读取慢曲匹配式的所有文件列表


#string_input_producer使用match_filenames_once获得的文件列表创建输入队列，将每个文件中每个文件加入队列中
#shuffle表示是否要打乱文件的顺序，一般会设置为True,这里为了演示队列对去顺序设置为False
filename_queue=tf.train.string_input_producer(files,shuffle=False)
# print(type(filename_queue))    #FIFOQueue对象

reader=tf.TFRecordReader()   #创建TFRecord文件读取对象

_,serialized_example=reader.read(filename_queue)   #read函数每次读取队列中一个文件(注意这里的文件都是序列化对象)

features=tf.parse_single_example(
    serialized_example,
    features={
    "i":tf.FixedLenFeature([],tf.int64),   #将原本的被变为feature的int型对象（i和j特征）再变回int型
    "j":tf.FixedLenFeature([],tf.int64),
})


with tf.Session() as sess:

    #match_filenames_once返回的文件列表作为临时变量并没有保存到checkpoint,所以不能用初始化全局变量的
    #global_variables_initializer来初始化，必须使用local_variables_initializer
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # print(sess.run(files))   #返回与match_filenames_once中相匹配的所有文件

    #产生线程的协调者并启动线程
    coord=tf.train.Coordinator()
    #不同于"7.3.1tf.QueueRuner和tf.Coordinator管理多线程队列操作"文件(以下简称原文件)中还需要定义
    #FIFOQueue，QueueRunner和add_queue_runner对象，原因如下:
    #1.string_input_producer返回的就是一个列表FIFOQueue对象
    #2.QueueRunner是线程运行对象，原文件定义该对象就是为了向一开始定义的空FIFOQueue中添加对象
    #  而这里通过string_input_producer返回的是一个已经有数据的FIFOQueue对象
    #3.原add_queue_runner对象是为了在执行start_queue_runners时，自动执行指定集合中的QueueRunner对象
    #  (默认启动tf.GraphKeys.QUEUE_RUNNERS集合中的所有QueueRunner),而这里其实也执行了tf.GraphKeys.QUEUE_RUNNERS集合中的所有QueueRunner对象
    #  只是集合中没有QueueRunner对象(至于为什么没有上两点已经解释了)
    #综上:这里只需要运行start_queue_runners获得线程对象即可

    threads=tf.train.start_queue_runners(sess=sess,coord=coord)

    for i in range(6):
        # 每次读取文件队列中的一个数据
        # 可能会问，两个文件中每个文件又两个数据，怎么能读出6个数据，是因为每次读取一个文件，当读完时候，又从头读取
        # 如果在string_input_producer函数中定义参数num_epochs=1，则表明文件近读取一轮,也就是4个
        # 那么这里如果在读6个就会报错
        print(sess.run([features["i"],features["j"]]))

    coord.request_stop()
    coord.join(threads)



