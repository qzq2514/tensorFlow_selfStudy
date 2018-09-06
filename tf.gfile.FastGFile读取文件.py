import tensorflow.gfile
import matplotlib.pyplot as plt
import tensorflow as tf


image_png=tf.gfile.FastGFile("temp.png","rb").read()


with tf.Session() as sess:
    image_png=tf.image.decode_png(image_png)
    print(sess.run(image_png))
    image_png=tf.image.convert_image_dtype(image_png,dtype=tf.uint8)

    plt.figure()
    plt.imshow(image_png.eval())
    plt.show()