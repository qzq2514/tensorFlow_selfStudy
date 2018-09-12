import tensorflow as tf

queue=tf.FIFOQueue(100,"float")   #定义一个大小是100，元素是float的队列


enqueue_op=queue.enqueue([tf.random_normal([1])])#定义队列的入队操作，入队一个长度为1的集合[1]

#tf.train.QueueRunner创建多个线程运行队列的入队操作，其第一个参数给出被操作的队列，[enqueue_op]*5表示需要启动五个线程
#每个线程运行的是enqueue_op操作(这里相当定义了五个线程，他们的target都是enqueue_op操作)
qr=tf.train.QueueRunner(queue,[enqueue_op]*5)


#将定一个过的QueueRunner加入计算图上指定的集合，在不指定集合情况下默认加入tf.GraphKeys.QUEUE_RUNNERS集合
tf.train.add_queue_runner(qr)

out_tensor=queue.dequeue()   #定义出队操作

with tf.Session() as sess:
    coord=tf.train.Coordinator()   #定义协同启动线程的协调者-Coordinator对象

    #使用QueueRunner，需要明确调用以下函数，在不指定集合情况下默认启动tf.GraphKeys.QUEUE_RUNNERS集合中的所有QueueRunner，所以start_queue_runners
    #和add_queue_runner会指定同一个集合，start_queue_runners返回集合中的线程，下句话运行后五个线程开始运行，即分别进行各自的target操作(这里都是入队操作)
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)

    # print(sess.run(queue.size()))    #队列通过五个线程各自的入队操作后已经包含多个个元素(重点是这里不一定就是5个元素，不清楚为什么？---
    #                                                                               因为循环读取，读完后又从头读取)
    # print(type(threads[2]))          #<class 'threading.Thread'>

    #将FIFOQueue队列进行三次出队操作
    for _ in range(3):print(sess.run(out_tensor)[0])    #获取队列中的值,因为入队的是一个长度为1的集合，可以使用[0]得到其中具体的元素值

    coord.request_stop()  #停止所有线程
    coord.join(threads)
    # print(sess.run(tf.random_normal([1])))



