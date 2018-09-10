import tensorflow as tf



que=tf.FIFOQueue(2,"int32")      #创建最多能装两个元素的队列，指定元素数据类型是int

init=que.enqueue_many(([0,10,],))          #enqueue_many初始化队列，将0,10元素入队

x=que.dequeue()      #出队

y=x+1       #出队元素加一

q_inc=que.enqueue(y)  #将相加后的元素入队

with tf.Session() as sess:
    init.run()
    for _ in range(5):
        v,_=sess.run([x,q_inc])   #重复执行出队元素加一后入队的操作
        print(v)

#依次输出值：0，10，1，11，2可以看到起到了队列的作用
