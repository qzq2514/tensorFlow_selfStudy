import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile('lena_full.jpg', 'rb').read()

#展示图片，im是解码后的三维矩阵
def showImage(im,num):
    plt.figure(num)
    plt.imshow(im.eval())
    plt.show()


with tf.Session() as sess:

# 图片的读写
    #decode_jpega将jpg文件解码为三维矩阵
    image_data=tf.image.decode_jpeg(image_raw_data)
    # print(image_data.eval())   #图片是一个三维矩阵

    showImage(image_data,0)

    #将图像的矩阵编码并保存成图片
    encode_image=tf.image.encode_jpeg(image_data)
    with tf.gfile.GFile("lena_output.jpg","wb") as f:
        f.write(encode_image.eval())

    #将图片转换成指定的类型，便于对图像进行处理操作，这部要放在保存之后，图像处理之前，因为保存的图片不能是tf.float32类型
    #后面的所有图像处理操作都是在图片解码后的三维矩阵上的操作
    image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)

#图像大小调整
    resized_image=tf.image.resize_images(image_data,[300,300],method=0)

    # method：
    # 0 双线性插值法（Bilinear interpolation）
    # 1 最近邻居法（Nearest neighbor interpolation）
    # 2 双三次插值法（Bicubic interpolation）
    # 3 面积插值法（Area interpolation）

    # print(image_data.get_shape())            #未调整大小后图片大小是未知的，(?, ?, ?)
    # print(resized_image.get_shape())         #调整后的图片则是指定的大小，(300, 300, ?)
    # showImage(resized_image,1)


    #还可以使用resize_image_with_crop_or_pad进行图像尺寸放缩，但是这里和上面resize_images不同，这里如果原始图像尺寸大于目标图像，那么就会截取原始
    #图像，如果小于那么周围用0填充
    croped=tf.image.resize_image_with_crop_or_pad(image_data,500,500)
    # showImage(croped,2)          #截出来的图有点色情，哈哈哈

    padded = tf.image.resize_image_with_crop_or_pad(image_data, 2000, 2000)
    # showImage(padded,3)

    #按比例进行截取，第二个参数是比例，范围为(0,1],下面就表示截取图片中间50%的部分
    central_cropped=tf.image.central_crop(image_data,0.5)
    # showImage(central_cropped, 4)



#图像翻转，在图像数据较少的情况下，可以起到上采样的作用
    fipped1=tf.image.flip_up_down(image_data)       #上下翻转
    # showImage(fipped1, 5)
    fipped2=tf.image.flip_left_right(image_data)    #左右翻转
    # showImage(fipped2, 6)
    transposed=tf.image.transpose_image(image_data) #对角线翻转
    # showImage(transposed, 7)
    fipped3=tf.image.random_flip_up_down(image_data)       #以一定的概率上下翻转，同样还有随机概率左右翻转
    # showImage(fipped3, 8)



#图像色彩调整
 #亮度调整
    adjusted1=tf.image.adjust_brightness(image_data,-0.5)   #亮度调低0.5
    # showImage(adjusted1,9)
    adjusted2=tf.image.adjust_brightness(image_data,-1.5)    #亮度调高0.5
    # showImage(adjusted2,10)
    random_adjusted3=tf.image.random_brightness(image_data,0.3)  #亮度调整范围在[-0.3,0.3)
    # showImage(random_adjusted3,11)

 #对比度调整
    adjusted11=tf.image.adjust_contrast(image_data,-5)          #对比度减少5
    # showImage(adjusted11,12)
    adjusted22=tf.image.adjust_contrast(image_data,5)           #对比度增加5
    # showImage(adjusted22,13)
    random_adjusted33=tf.image.random_contrast(image_data,0.1,0.3)  #指定区间范围内调整对比度,非负区间
    # showImage(random_adjusted33,14)

 #色相调整
    adjusted111=tf.image.adjust_hue(image_data,0.1)     #色相+0.1
    # showImage(adjusted111,15)
    random_adjusted333=tf.image.random_hue(image_data,0.3)  #在[0,0.3)范围内调整色相
    # showImage(random_adjusted333,16)

 #饱和度调整
    adjusted1111 = tf.image.adjust_saturation(image_data, -5)  # 饱和度减少5
    # showImage(adjusted1111,17)
    adjusted2222 = tf.image.adjust_saturation(image_data, 5)  # 饱和度增加5
    # showImage(adjusted2222,18)
    random_adjusted3333 = tf.image.random_saturation(image_data, 0.1, 0.3)  # 指定区间范围内调整饱和度,非负区间
    # showImage(random_adjusted3333,19)

 #图像标准化:图像三维矩阵的均值变为0，方差变为1
    norm_img=tf.image.per_image_standardization(image_data)   #不用per_image_whitening
    # showImage(norm_img,20)


#处理标注框
 #添加标注框
    batch_img=tf.expand_dims(resized_image,0)   #处理标注框的函数输入图像是一个batch的数据，也就是多个三维图像组成的思维矩阵没所以需要将解码后
                                             #的图像矩阵增加一维(同时，标注框的图像矩阵的数值要求是实数，所以需要使用convert_image_dtype函数
                                                             # 转为实数，这里之前已经转换过，不多处理,同时，最好使用尺寸较小的图像)
    boxes=tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]]) #每个数组是标注框的位置(Ymin,Xmin,Ymax,Xmax),这里给的都是比例下的相对位置
                                                                    #shape 为 [1,2,4]，1表示一组batch,2表示每个batch的两个图片，4表示位置
    result=tf.image.draw_bounding_boxes(batch_img,boxes)
    # plt.imshow(result[0].eval())            #不能直接使用plt.imshow(result.eval())
    # plt.show()

 #标注框截取图像
    #sample_distorted_bounding_box返回的三个变量，前两个用于tf.slice剪裁图像，最后的变量bbox_for_draw用于在draw_bounding_boxes中画出标注框
    #bbox_for_drawshape为 [1, 1, 4] ，第二个1表示每次只是随机生成一个标注框
    begin,size,bbox_for_draw=tf.image.sample_distorted_bounding_box(tf.shape(resized_image),bounding_boxes=boxes)

    # print(sess.run(begin))
    # print(sess.run(size))
    image_with_boxes=tf.image.draw_bounding_boxes(batch_img,bbox_for_draw)    #显示带有随机标注框的完整图片
    distorted_img=tf.slice(resized_image,begin,size)    #begin,size连个参数没太搞懂
    plt.subplot(121);plt.imshow(image_with_boxes[0].eval())
    plt.subplot(122);plt.imshow(distorted_img.eval())
    plt.show()

