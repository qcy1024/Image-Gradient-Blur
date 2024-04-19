from PIL import Image, ImageFilter
import numpy as np
from scipy.ndimage import gaussian_filter
import random
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

import numpy as np



def generate_gaussian_kernel(size, sigma):
    # 创建一个空的高斯核
    kernel = np.zeros(size)
    
    # 计算中心点的坐标
    center_x = size[0] // 2
    center_y = size[1] // 2
    
    # 生成高斯核
    for i in range(size[2]):  # 遍历通道
        for x in range(size[0]):  # 遍历行
            for y in range(size[1]):  # 遍历列
                # 计算当前位置到中心点的距离
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2)
                # 计算高斯值并将其赋给当前位置
                kernel[x, y, i] = np.exp(-distance)
    
    # 将中心元素设置为最大值
    kernel[center_x, center_y, :] = 1
    
    # 对每个通道的核进行归一化确保和为1
    for i in range(size[2]):
        kernel[:, :, i] /= np.sum(kernel[:, :, i])
    
    return kernel

image_path = "123.jpg"
sample_nums = 200

# 打开图片
image = Image.open(image_path)

image_matrix = np.array(image)
ti_image_matrix = ti.field(ti.u8,shape=image_matrix.shape)
ti_image_matrix.from_numpy(image_matrix)

# print(image_matrix)
# print("\n\n\n\n\n\n\n")
# print(ti_image_matrix)

ti_image_matrix_blurred = ti.field(ti.u8,ti_image_matrix.shape)
ti_image_matrix_blurred.from_numpy(image_matrix)

# 定义高斯核的大小和标准差
size = (33, 33, 3)  # 核的大小为 5x5，3个通道
sigma = 10

# 生成高斯核
# shape[0]:5, shape[1]:5, shape[2]:3
gaussian_kernel = generate_gaussian_kernel(size, sigma)

print("gaussian_kernel.shape: ",gaussian_kernel.shape)
# print(gaussian_kernel)
print("image_matrix.shape: ",image_matrix.shape)

ti_gaussian_kernel = ti.field(ti.f32,gaussian_kernel.shape)
ti_gaussian_kernel.from_numpy(gaussian_kernel)


@ti.kernel
def random_sample_gaussian_filter():
    print("ti_image_matrix.shape:",ti_image_matrix.shape)
    print("sample_nums:",sample_nums)

    for k,j,i in ti.ndrange(image_matrix.shape[0]-gaussian_kernel.shape[0],image_matrix.shape[1]-gaussian_kernel.shape[1],3):
        # if k == 0:
        #     print("k=0时，j = ",j)
        x_cen = k + ti_gaussian_kernel.shape[0] // 2
        y_cen = j + ti_gaussian_kernel.shape[1] // 2
        if x_cen == 0 or y_cen == 0:
            print("x_cen=",x_cen,"y_cen=",y_cen)
        t = 0.0

        for l in range(sample_nums):
            xi = int(ti.random() * ti_gaussian_kernel.shape[0])
            yi = int(ti.random() * ti_gaussian_kernel.shape[1])
            y = j + yi
            x = k + xi
            t += 0.0 + ti_image_matrix[x,y,i] / sample_nums * ti_gaussian_kernel[xi,yi,i] * ti_gaussian_kernel.shape[0] * ti_gaussian_kernel.shape[1] 
            if t > 255 :
                t = 255
            # elif t < 0 :
            #     t = 0
            # print("t=",t)
            # if k == 0 :
            #     print("j=",j,"t=",t)
            if j == 0 and t >=0 and t <= 255  :
                print("k=",k,"t=",t)

            ti_image_matrix_blurred[x_cen,y_cen,i] = ti.u8(t)
            
    #             image_matrix_blurred[x_cen,y_cen,i] = np.uint8(t)
    # image_matrix_blurred = image_matrix
    # for i in range(gaussian_kernel.shape[2]) :
    #     for j in range(image_matrix.shape[1]-gaussian_kernel.shape[1]):
    #         for k in range(image_matrix.shape[0]-gaussian_kernel.shape[0]):
    #             x_cen = k + gaussian_kernel.shape[1] // 2
    #             y_cen = j + gaussian_kernel.shape[0] // 2
    #             list_x = []
    #             list_y = []
    #             t = 0.0
    #             for l in range(sample_nums):
    #                 xi = random.randint(0,gaussian_kernel.shape[0]-1)
    #                 yi = random.randint(0,gaussian_kernel.shape[1]-1)
    #                 y = j + yi
    #                 x = k + xi
    #                 # print("x=",x)
    #                 # print("y=",y)
    #                 # print("k=",k)
                     
    #                 t += 0.0 + image_matrix[x,y,i] / sample_nums * gaussian_kernel[xi,yi,i] * gaussian_kernel.shape[0] * gaussian_kernel.shape[1] 
    #                 if t > 255 :
    #                     t = 255

                
    #             image_matrix_blurred[x_cen,y_cen,i] = np.uint8(t)
    # return image_matrix_blurred


  

    # blurred_image = Image.fromarray(image_matrix)
    # blurred_image.show()
    
    # # 保存模糊后的图片
    # output_path = "blurred_image.png"
    # blurred_image.save(output_path)
    # print("图片模糊处理完成，已保存为", output_path)





# random_sample_gaussian_filter()
# image_matrix_blurred = ti_image_matrix_blurred.to_numpy()
# print(image_matrix_blurred)
# blurred_image = Image.fromarray(image_matrix_blurred)
# blurred_image.save("./GSBlurred_image.png")    


