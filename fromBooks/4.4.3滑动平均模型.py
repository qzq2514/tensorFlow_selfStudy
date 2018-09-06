import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

v1=tf.Variable(0,dtype=tf.float32)   #滑动平均的初始变量值为0
step=tf.Variable(0,trainable=False)

#定义衰减率为0.99
ema=tf.train.ExponentialMovingAverage(0.99,step)

#添加/更新变量，为之维护影子变量，每次apply更新就会自动训练一遍模型
maintrain_averages_op=ema.apply([v1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run([v1,ema.average(v1)]))

    sess.run(tf.assign(v1,5))

    #进行一次滑动平均,衰减率为min{0.99,(1+step)/(10+step)=0.1}=0.1
    sess.run(maintrain_averages_op)
    #v1的滑动平均会被更新成0.1*0+0.9*5=4.5
    print(sess.run([v1, ema.average(v1)]))   #average()取出影子变量的值


    sess.run(tf.assign(step,10000))   #衰减率变成了 min{0.99,(1+step)/(10+step)~0.999}=0.99
    sess.run(tf.assign(v1,10))

    sess.run(maintrain_averages_op)  #v1的影子变量为第22行获得的值-4.5，新值就是被重新assign的10，这里通过调用apply进行一次滑动平均，
                                     # 滑动平均中的v1=0.99*4.5+0.01*10=4.555(又变成下一册滑动平均的影子值)
    print(sess.run([v1,ema.average(v1)]))

    #滑动平均模型改变的是模型中的变量的影子变量,像这里，滑动平均模型中一开始通过apply函数添加v1的影子模型
    #之后再每次apply改变的都是v1的影子模型，v1本身不会变


