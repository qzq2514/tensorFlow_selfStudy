from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x=mnist.train.images
y=mnist.train.labels
#
arr_img=x[0].reshape([28,28])   #得到并reshape下标为1的图片
# arr_img=np.array([[0, 0, 0, 1, 0, 0, 0],
#                   [0, 0, 0, 1, 0, 0, 0],
#                   [0, 0, 0, 1, 0, 0, 0],
#                   [0, 0, 0, 1, 0, 0, 0],
#                   [0, 0, 0, 1, 0, 0, 0],
#                   [0, 0, 0, 1, 0, 0, 0],])
# print(arr_img)
# print(y[1])
fig = plt.figure()
plt.imshow(arr_img,cmap='gray')
plt.show()

