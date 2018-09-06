import tensorflow as tf
import matplotlib.pyplot as plt


learnin_rate=0.1
decay_rate=0.96      #衰减速率，即每一次学习都衰减为原来的0.96
global_steps=1000    #总学习次数


decay_steps=100

global_=tf.placeholder(dtype=tf.int32)

#staircase=True表示每decay_steps更新一次decay_rate，即每decay_steps轮，学习率就乘以decay_rate一次得到新的学习率
#staircase=Flase表示每次都更新一次decay_rate(默认)，即每次global_变化，都会导致decay_rate的指数-(global_ / decay_steps)变化一次
#简单点说就是staircase=True时，global_ / decay_steps是整除，staircase=Flase是实数相除

#decayed_learning_rate = learning_rate * decay_rate ^ (global_ / decay_steps)
#global_表示当前的迭代轮数，是不断变化的
c=tf.train.exponential_decay(learnin_rate,global_,decay_steps,decay_rate,staircase=True)
d=tf.train.exponential_decay(learnin_rate,global_,decay_steps,decay_rate,staircase=False)

T_C=[]
F_D=[]


with tf.Session() as sess:
    for i in range(global_steps):
        T_c=sess.run(c,feed_dict={global_:i})
        T_C.append(T_c)
        F_d=sess.run(d,feed_dict={global_:i})
        F_D.append(F_d)


#这里F_D和T_C的长度都是一样的，只是T_C是其中每decay_steps更新一次
plt.figure(1)
l1,=plt.plot(range(global_steps),F_D,"r-")
l2,=plt.plot(range(global_steps),T_C,"b-")

plt.legend(handles=[l1,l2],labels=['staircase=False', 'staircase=True'],loc="best")
plt.show()


