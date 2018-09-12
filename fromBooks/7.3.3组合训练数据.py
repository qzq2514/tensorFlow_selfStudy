import tensorflow as tf

# 改写"7.3.2TFRecord样例使用"中每次读取一个数据
#本程序将多个数据保存成一个batch，供神经网络训练使用，加入这里的"i"特征极为输入向量(如一张图像的像素矩阵)
#"j"特征即为数据的标签


files=tf.train.match_filenames_once("TFRecord_Dir/threadLearning/data.tfrecord-*")  #读取慢曲匹配式的所有文件列表


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

#--------------------到此为止和"7.3.2TFRecord样例使用"的前半部分是一样的

example,label=features["i"],features["j"]   #每次获一个数据

batch_size=3  #batch大小
capacity=1000+3*batch_size  #队列的最大容量


#tf.train.batc将单个数据入队，而将一个batch出队，所以每次会得到指定大小的batch
#相同的还有tf.train.shuffle_batch函数，和tf.train.batch类似，但是是随机将样例组合成batch
#同时其还有个min_after_dequeue表示每次出队的batch中最少的数据量，因为太少的数据量，打乱是没意义的
#所以当数据量过少时，该函数的出队操作会等待数据的入队操作，直到有足够的数据(至少有min_after_dequeue数据才组合成batch)
example_batch,label_batch=tf.train.batch([example,label],batch_size=batch_size,
                                         capacity=capacity)


with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())

    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)


    for i in range(3):   #打印三个bathc
        cur_example_batch,cur_label_batch=sess.run([example_batch,label_batch])
        print(cur_example_batch,cur_label_batch)  #这里可以看到每个batcn的输入特征("i")和
        print("-----------")                      #标签("j")都被放入了不同的两个集合

    coord.request_stop()
    coord.join(threads)