import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with tf.variable_scope("foo"):    #定义名为"foo"的命名空间
    #tf.get_variable和tf.Variable()功能一样，生成变量
    #但是tf.get_variable在定义时必须给定名称(name)，然后给定shape和初始化函数
    v=tf.get_variable(name="v",shape=[1],initializer=tf.constant_initializer(25.14))


# with tf.variable_scope("foo"):      #在同一个命名空间中无法使用get_variable产生两个同名的变量,所以这句话会报错
#     v=tf.get_variable(name="v",shape=[1])



with tf.variable_scope("foo",reuse=True):        #如果想要获取之前已经定义好的变量，必须设定参数reuse=True
    v1 = tf.get_variable(name="v")
    print(v1==v)      #打印出来True



with tf.variable_scope("bar",reuse=True):        #在reuse=True的命名空间只能获取该命名空间已经定义的变量
    # v1 = tf.get_variable(name="v")              #因为这里连变量的初始化都没有，所以只是获取名为"v"的变量，或报错

    #注意下句即便初始化也不行，因为在reuse=True的命名空间只能获取变量，不能创建
    # v=tf.get_variable(name="v",shape=[1,2],initializer=tf.constant_initializer([25.14,6.19]))
    print(v)


print("---------------------------------")

with tf.variable_scope("root"):
    print(tf.get_variable_scope().reuse)           #输出当前命名空间的reuse,这里是False

    with tf.variable_scope("foo",reuse=True):
        print(tf.get_variable_scope().reuse)       #输出True

        with tf.variable_scope("bar"):             #这里新嵌套的内层命名空间没有指定reuse,则其和外面一层保持一致,输出True
            print(tf.get_variable_scope().reuse)

    print(tf.get_variable_scope().reuse)           #输出False


print("---------------------------------")


#定义矩阵时，shape要么写在name参数后，要么写在initializer参数里面,常数初始化器默认值全为0
# a1=tf.get_variable(name="a",shape=[3,4],initializer=tf.constant_initializer())
a1=tf.get_variable(name="a",initializer=tf.constant([[1,2,3]]))

print(a1.name)   #输出"a:0"

with tf.variable_scope("foo"):
    b1=tf.get_variable("b",initializer=tf.truncated_normal([2,3]))
    print(b1.name)         #输出foo/b:0 形如：命名空间/变量自身名
    with tf.variable_scope("bar"):
        a2=tf.get_variable("a",shape=[2])
        print(a2==a1)       #False,命名空间不同
        print(a2.name)      #输出foo/bar/a:0,构成变量的总体name(注意命名空间可以嵌套)

with tf.variable_scope("",reuse=True):
    a3=tf.get_variable("foo/bar/a")  #可以通过带有命名空间的变量名获取其他命名空间的变量
    print(a3 == a2)     #True
    print(a3 == a1)     #False       这里a3,a1在不同的命名空间，不相等
    print(a3.name)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("a3:",sess.run(a3))
        print("a1:", sess.run(a1))


print("------------------------")
m=tf.Variable(tf.constant(3,shape=[3,4]))     #定义全为3的矩阵
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(m))
