import tensorflow as tf
import numpy as np
import threading
import time

def myLoop(coord,worker_id):
    while not  coord.should_stop():    #should_stop()表明，用于表示Coordinator其内的线程是否应该停止
        if np.random.rand()<0.1:
            print("Stoping from id:%d"%worker_id)
            coord.request_stop()    #调用容器的request_stop来通知其内的所有线程停止运行
        else:
            print("Working on id:%d"%worker_id)
        time.sleep(1)


coord=tf.train.Coordinator()    #Coordinator相当于一个盛放线程的容器(或者叫协调者)，对于Coordinator的改变可能会影响其内的线程
threads=[threading.Thread(target=myLoop,args=(coord,i)) for i in range(5)]

for t in threads:
    t.start()              #启动所有线程

coord.join(threads)       #将所有的线程放入Coordinator中



#又是可能会出现已经在某线程运行时就已经执行了request_stop，但是后面还是会有线程继续执行并输出"Working on id:x"
#这是线程是并行运行，虽然在某个线程内执行了request_stop(),改变should_stop()值，但是由于并行运行，其他的线程可能已经执行了判断容器
#的should_stop()操作，这时判断为True就输出"Working on id:x"但是在暂停一段时间后，开始下一轮运行时，就能捕捉到这时候should_stop()
#已经被其他线程改变为False了，就循环结束，线程调用函数结束