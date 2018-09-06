import tensorflow as tf

#保存模型
# v=tf.Variable(0,dtype=tf.float32,name="v")
#
# #在没有申请滑动平均模型时，此时只有一个变量
# for var in tf.global_variables():
#     print(var.name)
#
# print("-------------")
# ema=tf.train.ExponentialMovingAverage(0.1)
# mainTrain_average_op=ema.apply(tf.global_variables())
#
# #在使用平均滑动模型并apply后,就多了一个变量v/ExponentialMovingAverage:0
# for var in tf.global_variables():
#     print(var.name)
#
# print("-------------")
# saver=tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#
#
#     sess.run(tf.assign(v,10))
#     sess.run(mainTrain_average_op)   #进行一次滑动平均
#
#     saver.save(sess,"Saver/MovingAverage")
#     print(sess.run([v,ema.average(v)]))


#加载模型方法一:手动给定Saver参数，读取保存好的滑动平均模型参数
# vv=tf.Variable(0,dtype=tf.float32,name="vv")
# saver=tf.train.Saver({"v/ExponentialMovingAverage":vv})
#
# with tf.Session() as sess:
#     saver.restore(sess,"Saver/MovingAverage")
#     print(sess.run(vv))     #9.0

vvv=tf.Variable(0,dtype=tf.float32,name="v")
ema=tf.train.ExponentialMovingAverage(0.1)
print((ema.variables_to_restore()))   #自动生成类似上边Saver的字典参数
                        #{'v/ExponentialMovingAverage': <tf.Variable 'v:0' shape=() dtype=float32_ref>}
                        #注意此时变量的name就不能随便写了，不然会找不到变量的影子变量
saver=tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess,"Saver/MovingAverage")
    print(sess.run(vvv))     #根据name的对应，vvv被赋值

