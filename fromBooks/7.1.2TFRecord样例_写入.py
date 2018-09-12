import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#TFRecode是统一的输入数据格式，被保存成字典形式
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

mmnist=input_data.read_data_sets("MNIST_data",dtype=tf.uint8,one_hot=True)


images=mmnist.train.images
labels=mmnist.train.labels
pixels=images.shape[1]          #像素是784，也就是28*28
print(pixels)

num_examples=mmnist.train.num_examples    #也就是images.shape[0]
print(num_examples)

#显示第一个mnist数据
# image1=images[1]
# print(image1.shape)     #(784,)
# image1=image1.reshape([28,28])
# plt.figure()
# plt.imshow(image1,cmap="gray")
# plt.show()


#开始写入

#TFRecord可以将数据规定成统一的数据格式，便于后面对于数据的处理
fileName="TFRecord_Dir/output.tfrecords"
writer=tf.python_io.TFRecordWriter(fileName)

for ind in range(num_examples):
    #将图像矩阵转化为一个字符串
    image_raw=images[ind].tostring()
    # print(image_raw)
    #讲一个样例转化为tf.train.Features类型的Protocol Buffer,并将该信息写入这个数据结构
    example=tf.train.Example(features=tf.train.Features(feature={
        "pixels":_int64_feature(pixels),
        "label":_int64_feature(np.argmax(labels[ind])),     #TFRecord数据属性的整型使用_int64_feature转换为合适类型
        "image_raw":_bytes_feature(image_raw)
    }))
    if ind<10:print(np.argmax(labels[ind]))
    #将一个example变成序列化写入TFRecord文件
    writer.write(example.SerializeToString())

writer.close()




