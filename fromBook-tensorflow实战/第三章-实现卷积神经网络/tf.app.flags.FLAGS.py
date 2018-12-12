import tensorflow as tf
import argparse
import sys
import time
from datetime import datetime

#1.tensorflow中添加命令行中的可选参数(参数名，参数默认值，参数提醒)
# FLAGS=tf.app.flags.FLAGS
# tf.app.flags.DEFINE_float('flag_float',0.01,'input a float')
# tf.app.flags.DEFINE_integer('flag_int',400,'input a int')
# tf.app.flags.DEFINE_boolean('flag_bool',True,'input a bool')
# tf.app.flags.DEFINE_string('flag_string','yes','input a string')

# #可直接通过FLAGS.得到命令行参数
# print("------tensorflow参数------")
# print(FLAGS.flag_float)
# print(FLAGS.flag_int)
# print(FLAGS.flag_bool)
# print(FLAGS.flag_string)


#2.python原生获取命令行参数-不要与tf.app.flags.FLAGS混着用
# argParse=argparse.ArgumentParser()
# argParse.add_argument("-p", "--path", required=True,help="path to input image")
# # argParse.add_argument('-b',"--argBool",required=True)
# args=vars(argParse.parse_args())
# print(args["path"])


#3.实现清屏的时间和进度条的动态打印
#使用了转移字符"\r"使得光标回到行首，再把缓冲区显示出来
for i in range(51):
    a = datetime.now()
    sys.stdout.write("\r{0}--->{1}{2}".format(a,"."*i,"%.2f%%"%(i*2)))
    sys.stdout.flush()
    time.sleep(1)