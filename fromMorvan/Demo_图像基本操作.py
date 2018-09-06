from PIL import Image
import numpy as np

arr_img=np.array(Image.open("scene.jpg"))
# arr_img=np.array([[0, 0, 0, 1, 0, 0, 0],
#                   [0, 0, 0, 1, 0, 0, 0],
#                   [0, 0, 0, 1, 0, 0, 0],
#                   [0, 0, 0, 1, 0, 0, 0],
#                   [0, 0, 0, 1, 0, 0, 0],
#                   [0, 0, 0, 1, 0, 0, 0],])
print(arr_img.shape,arr_img.dtype)
im=Image.fromarray(arr_img)  #根据像素矩阵生成图像
im.show()      #展示图像






