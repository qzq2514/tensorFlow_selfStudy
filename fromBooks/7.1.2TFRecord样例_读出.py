import tensorflow as tf

reader=tf.TFRecordReader()

#创建一个队列来维护输入文件列表
fileName_queue=tf.train.string_input_producer(["TFRecord_Dir/output.tfrecords"])

#读取一个样例，也可以使用read_up_to函数一次性读取多个样例
_,serialized_example=reader.read(fileName_queue)

#解析一个样例，也可以使用parse_example解析多个样例
features=tf.parse_single_example(
            serialized_example,
            features={
                #Tensorflow两种属性解析方法：
                #1.FixedLenFeature,得到Tensor
                #2.VarLenFeature,得到SparseTensor,用于处理稀疏矩阵
                "image_raw":tf.FixedLenFeature([],tf.string),
                "pixels":tf.FixedLenFeature([],tf.int64),
                "label":tf.FixedLenFeature([],tf.int64),
            })


#tf.decode_raw将字符串解析成图像对应的像素数组
images=tf.decode_raw(features["image_raw"],tf.uint8)
images_=tf.reshape(images,[28,28,-1])       #将原向量形式的像素数据转变为矩阵形式
labels=tf.cast(features["label"],tf.int32)
pixels=tf.cast(features["pixels"],tf.int32)


sess=tf.Session()
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(sess=sess,coord=coord)


#一次读取十个样例,样例读完后从头读取
for i in range(10):
    image,image_,label,pixel=sess.run([images,images_,labels,pixels])
    print(image.shape)   #mnist数据集像素是展开的-长度为784的向量
    print(image_.shape)
    print(label)
