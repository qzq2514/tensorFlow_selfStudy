import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=False)
#one_hot=False此时标签y_大小是[batcn_size,]每一个元素属于[0,num) num是类别数


x = tf.placeholder(tf.float32, [None, 784])
y_=tf.placeholder(tf.int32,[None,])

dense1 = tf.layers.dense(inputs=x,
                      units=1024,
                      activation=tf.nn.relu)
dense2= tf.layers.dense(inputs=dense1,
                      units=512,
                      activation=tf.nn.relu,)

logits= tf.layers.dense(inputs=dense2,
                        units=10,
                        activation=None)


#这一句和下面两句起到共同的效果，且sparse_softmax_cross_entropy函数和
#sparse_softmax_cross_entropy_with_logits函数中的labels参数都是[batch_size,]大小
#logits参数都是[batch_size,num]大小

# loss=tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=logits)
loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_,logits=logits)  #返回[batch_size,]大小的loss
loss=tf.reduce_mean(loss)
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess=tf.Session()
sess.run(tf.global_variables_initializer())

is_train=True
saver=tf.train.Saver(max_to_keep=3)    #最多保存当前的三个模型(保存太多，浪费内存)

#训练阶段，训练参数并保存参数
if is_train:
    max_acc=0
    f=open('ckpt/acc.txt','w')
    for i in range(100):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(train_op, feed_dict={x: batch_xs, y_: batch_ys})
      val_loss,val_acc=sess.run([loss,acc], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
      print('epoch:%d, val_loss:%f, val_acc:%f'%(i,val_loss,val_acc))
      f.write(str(i+1)+', val_acc: '+str(val_acc)+'\n')    #将精度写入txt文件中
      if val_acc>max_acc:
          max_acc=val_acc
          saver.save(sess,'ckpt/mnist.ckpt',global_step=i+1)
    f.close()

#验证阶段，提取参数
else:
    model_file=tf.train.latest_checkpoint('ckpt/')
    saver.restore(sess,model_file)
    val_loss,val_acc=sess.run([loss,acc], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print('val_loss:%f, val_acc:%f'%(val_loss,val_acc))
sess.close()