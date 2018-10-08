from sklearn import cross_validation
from sklearn import datasets
from sklearn import  metrics
import tensorflow as tf

learn=tf.contrib.learn


def my_model(features,target):
    target=tf.one_hot(target,3,1,0)    #target是向量或矩阵，3表示每个标签的one-hot向量长度，
                                       #第三、四个参数分别表示on_value、off_value，默认值分别为1、0
                                       #[0,1,2]--->[[ 1.,  0.,  0.],[ 0.,  1.,  0.],[ 0.,  0.,  1.]]

    #logistic_regression封装了一个单层全连接神经网络
    logits,loss=learn.models.logistic_regression(features,target)
    #创建模型的优化器，并得到优化步骤
    train_op=tf.contrib.layers.optimize_loss(
        loss,                                   #损失函数
        tf.contrib.framework.get_global_step(), #获取训练步骤并在训练时更新
        optimizer="Adagrad",                    #优化器类型
        learning_rate=0.1)                      #学习率

    print("logits.shape:",logits.shape)
    return tf.arg_max(logits,1),loss,train_op   #返回值要将logits进行arg_max，以便后面计算精确度时保证与target形状一致


iris=datasets.load_iris()
# print(iris.data.shape)                     #鸢尾花数据集有150个样本，每个样本有4个输入特征
# print(iris.target.shape)
x_train,x_test,y_train,y_test=cross_validation.train_test_split(iris.data,iris.target,test_size=0.2,random_state=0)#2比8划分测试集和训练集

classifier=learn.Estimator(model_fn=my_model)    #封装好定义的模型
classifier.fit(x_train,y_train,steps=500)        #使用封装好的模型和训练数据执行500轮训练,这会将fit函数的前两个参数x_train,y_train作为Estimator的model_fn指定的模型的参数

y_predicted=classifier.predict(x_test)
print(type(y_predicted))             #返回generator，要将其变为list
y_predicted=list(y_predicted)

score=metrics.accuracy_score(y_test,y_predicted)

print("精确度为:%.2f%%",(score*100))

