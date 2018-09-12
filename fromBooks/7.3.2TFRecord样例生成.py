import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


#模拟将大量的数据写入不同的TFRecord文件,
num_shards=2    #总TFRecord文件数
instance_per_shards=2   #每个TFRecord文件的数据数目

for i in range(num_shards):
    # 定义每个TFRecord文件的文件名(%.5d:长度为5的整形，用.补充),i表示当前的第i个文件数，
    # num_shards表示文件数(这种形式便于后面使用正则表达式获取文件)
    # 形如"data.tfrecord-00000-of-00002"
    filename=("TFRecord_Dir/threadLearning/data.tfrecord-%.5d-of-%.5d"%(i,num_shards))
    writer=tf.python_io.TFRecordWriter(filename)
    for j in range(instance_per_shards):
        example=tf.train.Example(features=tf.train.Features(feature={
            "i":_int64_feature(i),
            "j":_int64_feature(j)    #TFRecord数据属性的整型使用_int64_feature转换为合适类型
        }))
        writer.write(example.SerializeToString())     #将一个数据写入指定的TFRecord文件文件中，这里的数据很简单，就两个属性:"i"和"j"

    writer.close()



