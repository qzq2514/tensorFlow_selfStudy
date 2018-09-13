import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


a=tf.constant([1.0,2.0],name="a")
b=tf.constant([2.1,3.1],name="b")

result=a+b

print(tf.Session().run(result))