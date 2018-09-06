import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile


Bottleneck_tensor_node=2048   #瓶颈层(即原模型除最后一层全连接前的所有网络层)的结点数

Bottleneck_tensor_name="pool_3/_reshape:0"   #瓶颈层结果的张量名,使用tensor.name获得

Jpeg_data_tensor_name="DecodeJpeg/contents:0"  #图像输入张量对应的名字

#模型所在文件夹
Model_dir="inception_dec_2015"

Model_file="tensorflow_inception_graph.pb"

Cache_dir="temp/bottleneck"   #临时文件保存位置

Input_data="flower_photos"


Validation_percentage=10    #验证集数据百分比

Test_percentage=10

Learnin_rate=0.01
Steps=10
Batch=100

def creat_image_lists(test_per,validation_per):
    result={}

    #os.walk返回的是一个3元素元组构成的集合 (root, dirs, files) ，分别表示遍历的完整路径名，该路径下的目录列表和该路径下文件列表
    sub_dirs=[x[0] for x in os.walk(Input_data)]   #得到指定目录下的子目录名,这里返回只得到三元组的第一个，即目录：
                                                    #flower_photos
                                                    #flower_photos / roses
                                                    #flower_photos / sunflowers
                                                    #flower_photos / daisy
                                                    #flower_photos / dandelion
                                                    #flower_photos / tulips

    #得到的第一个是当前目录，这里即表示flower_photos目录名
    is_root_dir=True

    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir=False
            continue
        # print(sub_dir)
        extensions=["jpg","jpeg","JPG","JPEG"]
        file_list=[]
        dir_name=os.path.basename(sub_dir)         #返回path最后的文件名或者目录名,如何path以／或\结尾，那么就会返回空值，这里返回roses，sunflowers等
        for extension in extensions:
            file_glob=os.path.join(Input_data,dir_name,"*."+extension) #拼接多个路径：flower_photos/roses/*.jpg，这里构成的匹配模式
            file_list.extend(glob.glob(file_glob))  #glob.glob根据正则表达式返回所有符合正则表达式的文件或者目录的路径，这里是获得当前种类花图片的所有路径
                                                    #list的extend方法在列表末尾一次性追加另一个序列中的多个值，这里将dir_name种类内所有花的路径放在file_list中
        if not file_list:continue     #目录为空，直接continue

        #到这里file_list存放当前种类(dir_name)所有花的完整路径

        label_name=dir_name.lower()    #获取当前花的种类(全部小写形式)

        training_images=[]
        testing_images=[]
        validation_images=[]
        for file_name in file_list:    #根据测试集，验证集，训练集比重将图片分配至不同集合
            base_name=os.path.basename(file_name)     #得到当前花图片的文件名
            chance=np.random.randint(100)
            if chance <validation_per:
                validation_images.append(base_name)
            if chance<(validation_per+test_per):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        result[label_name]={
            "dir":dir_name,              #某个类的上一级目录： tulips，也就是花的种类
            "training":training_images,
            "testing":testing_images,
            "validation":validation_images,
        }

    return result   #result是一个字典集合，存放不同类别的数据集

#根据类别名称，所属数据集和图拼图编号(index)获取一张图片的地址
def get_image_path(image_lists,image_dir,label_name,index,category):
    label_lists=image_lists[label_name]
    category_list=label_lists[category]         #获取label_name该类别下的category(如"training")数据集
    mod_index=index%len(category_list)
    base_name=category_list[mod_index]          #获取该数据集下取余后第index个图片名称
    sub_dir=label_lists["dir"]

    full_path=os.path.join(image_dir,sub_dir,base_name)  #image_dir可能是临时目录，如："temp/bottleneck"，也可能是主文件目录一级目录：flower_photos
    return full_path                                     #sub_dir是图片父目录，如:rose,而base_name是文件名

def get_bottleneck_path(image_lists,label_name,index,category):
    return get_image_path(image_lists,Cache_dir,label_name,index,category)+".txt"    #完整的图片路径加上后缀名".txt"，这是瓶颈层数据文件的保存格式

def run_bottleneck_on_image(sess,image_data,image_data_tensor,bottleneck_tensor):
    #根据inceptionV3模型计算现在输入数据为image_data时的瓶颈层的输出向量bottleneck_tensor
    bottleneck_values=sess.run(bottleneck_tensor,feed_dict={image_data_tensor:image_data})
    bottleneck_values=np.squeeze(bottleneck_values)
    return bottleneck_values



#获取已经保存下来的特征向量，如果没保存，那么就建立新的
#最后返回瓶颈层的特征向量
def get_or_create_bottleneck(sess,image_lists,label_name,index,category,jpeg_data_tensor,bottleneck_tensor):
    label_lists=image_lists[label_name]    #根据标签名(如:rose,sunflowers)获得该类花的数据集
    sub_dir=label_lists["dir"]      #某个类较为完整的路径名：flower_photos / tulips
    sub_dir_path=os.path.join(Cache_dir,sub_dir)   #获取已经保存下载的临时文件的路径，Cache_dir="temp/bottleneck"
    if not os.path.exists(sub_dir_path):os.makedirs(sub_dir_path)   #没有该目录则创建

    bottleneck_path=get_bottleneck_path(image_lists,label_name,index,category)    #获取某个图片的特征文件


    if not os.path.exists(bottleneck_path):     #某个图片的特征文件不存在

        #Input_data="flower_photos"
        image_path=get_image_path(image_lists,Input_data,label_name,index,category)
        image_data=gfile.FastGFile(image_path,"rb").read()    #读取图片文件

        #通过inceptionV3模型计算在image_data图片作为整体的网络输入层时，在瓶颈层的特征向量
        bottleneck_values=run_bottleneck_on_image(sess,image_data,jpeg_data_tensor,bottleneck_tensor)

        #将计算得到的image_data图片经过inceptionV3模型后在瓶颈层的特征向量保存起来
        bottleneck_string=",".join([str(x) for x in bottleneck_values])

        # print(bottleneck_string)
        with open(bottleneck_path,"w") as bottleneck_file:
            bottleneck_file.write(bottleneck_string)    #将特征向量保存起来

    else:
        with open(bottleneck_path, "r") as bottleneck_file:
            bottleneck_string=bottleneck_file.read()  # 将特征向量保存起来


        bottleneck_values=[float(x) for x in bottleneck_string.split(",")]

    return bottleneck_values

#category是字符串形式，如"training",how_many是Batch数
#得到how_many个样本在瓶颈层的特征向量bottlenecks，和他们的标签
def get_random_cache_bottleneck(sess,n_class,image_lists,how_many,category,jpeg_data_tensor,bottleneck_tensor):
    bottlenecks=[]
    ground_truths=[]

    for _ in range(how_many):    #产生how_many个样本
       label_index=random.randrange(n_class)    #randrange(start,stop,step)可以指定增量，randint(start,stop)增量只能为1，但这里没啥区别，都是前闭后开区间
       label_name=list(image_lists.keys())[label_index]    #得到随机的某个类的图片label_name是roses，sunflowers等字符串
       image_index=random.randrange(65536)
       bottleneck=get_or_create_bottleneck(sess,image_lists,label_name,image_index,category,jpeg_data_tensor,bottleneck_tensor)
       ground_truth=np.zeros(n_class,dtype=np.float32)
       ground_truth[label_index]=1.0      #属于第label_index类
       bottlenecks.append(bottleneck)
       ground_truths.append(ground_truth)

    return bottlenecks,ground_truths

def get_test_bottlenecks(sess,image_lists,n_class,jpeg_data_tensor,bottleneck_tensor):
    bottlenecks=[]
    ground_truths=[]
    label_name_list=list(image_lists.keys())

    for label_index,label_name in enumerate(label_name_list):    #把list放在enumerate中遍历，可以多一个遍历参数label_index，从0自增
        category="testing"
        for index,unused_base_name in enumerate(image_lists[label_name][category]):
            bottleneck=get_or_create_bottleneck(sess,image_lists,label_name,index,category,
                                                jpeg_data_tensor,bottleneck_tensor)
            ground_truth=np.zeros(n_class,dtype=np.float32)
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)

    return bottlenecks,ground_truths

def main(argv=None):
    image_lists=creat_image_lists(Test_percentage,Validation_percentage)
    n_class=len(image_lists.keys())   #获得类别数
    with gfile.FastGFile(os.path.join(Model_dir,Model_file),"rb") as f:   #gfile.FastGFile实现对文件的读取，这里读取计算图的pb文件
        graph_def=tf.GraphDef()
        graph_def.ParseFromString(f.read())

    bottleneck_tensor,jpeg_data_tensor=tf.import_graph_def(         #import_graph_def得到对应计算图的指定张量
            graph_def,return_elements=[Bottleneck_tensor_name,Jpeg_data_tensor_name]
        )

    #瓶颈层的输入
    bottleneck_input=tf.placeholder(tf.float32,[None,Bottleneck_tensor_node],name="BottleneckInputPlaceholder")

    ground_truth_input=tf.placeholder(tf.float32,[None,n_class],name="GroundTruthInput")

    with tf.name_scope("Final_train_op"):
        #瓶颈层和输出层添加一层全连接层，而且weights和biases才是本迁移学习小程序真正需要训练的参数
        weights=tf.Variable(tf.truncated_normal([Bottleneck_tensor_node,n_class],stddev=0.001))

        biases=tf.Variable(tf.zeros([n_class]))

        logits=tf.matmul(bottleneck_input,weights)+biases  #得到输出层
        final_tensor=tf.nn.softmax(logits)                 #对输出层进行softmax

        #logits-(m,n_class)  ,ground_truth_input-(m,n_class)
        #sparse_softmax_cross_entropy_with_logits函数的labels参数才应该是(m,)
        #以下的函数中logits和labels是尺寸一样的，对logits先进行softmax，然后取对数，之后和labels逐位相乘
    cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=ground_truth_input)
    train_step=tf.train.GradientDescentOptimizer(Learnin_rate).minimize(cross_entropy)

    #计算正确率
    with tf.name_scope("evaluation"):
        correct_pred=tf.equal(tf.argmax(final_tensor,1),tf.argmax(ground_truth_input,1))
        evaluation_step=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #开始训练
        for i in range(Steps):
            #image_lists是字典形式的数据集，里面存放的是图片文件的路径
            #bottleneck_tensor,jpeg_data_tensor是从incetionV3计算图中获取的瓶颈层和输入层的张量
            #获取指定Batch数据样本
            train_bottlenecks,train_ground_truth=get_random_cache_bottleneck(sess,n_class,image_lists,Batch,
                                                                            "training",jpeg_data_tensor,bottleneck_tensor)

            sess.run(train_step,feed_dict={bottleneck_input:train_bottlenecks,ground_truth_input:train_ground_truth})

            if i %100==0 or i+1==Steps:
                validation_bottlenecks,validation_ground_truth=get_random_cache_bottleneck(sess,n_class,image_lists,Batch,
                                                                                           "validation",jpeg_data_tensor,bottleneck_tensor)
                validation_accur=sess.run(evaluation_step,feed_dict={bottleneck_input:validation_bottlenecks,
                                                                     ground_truth_input:validation_ground_truth})
                print("经过%s步后，在%s个随机样本构成的验证集上的精确度为:%s%%"%(i,Batch,validation_accur*100))


        test_bottlenecks,test_ground_truth=get_test_bottlenecks(sess,image_lists,n_class,jpeg_data_tensor,bottleneck_tensor)
        test_accur=sess.run(evaluation_step,feed_dict={bottleneck_input:test_bottlenecks,
                                                       ground_truth_input:test_ground_truth})
        print("最终模型在测试集上的精确度为:%s%%"%(test_accur*100))

if __name__ == '__main__':
    tf.app.run()

#可以运行，但是超级浪费CPU