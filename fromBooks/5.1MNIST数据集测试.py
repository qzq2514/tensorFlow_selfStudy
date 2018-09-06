from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

#基本属性
print("训练集大小%s" % mnist.train.num_examples)

print("验证集大小%s" % mnist.validation.num_examples)

print("测试集大小%s" % mnist.test.num_examples)

print("训练集大小%s" % mnist.train.num_examples)

print("第一个训练集数据:",mnist.train.images[0].shape)   #(784,)

print("第一个训练集数据标签:",mnist.train.labels[0])

#可视化
print("-----------------")
X=mnist.train.images
Y=mnist.train.labels
img=X[0].reshape([28,28])

fig=plt.figure()
plt.imshow(img,cmap="gray")    #cmap用于设置热图的Colormap
plt.show()



#next_batch不断选择下一个batch,在随机梯度下降算法中经常用到，横样本
batch_size=100
xs,ys=mnist.train.next_batch(batch_size)
print(xs.shape)
print(ys.shape)