from PIL import Image
import numpy as np
import taichi as ti
from pathlib import Path
import time
ti.init(arch=ti.gpu)

image_path = "ImageGradientBlur/123.jpg"
out_folder = "out"
half_kernel_size = 10
sigma = 10
gs_kernel_size = 4 * half_kernel_size + 1
gs_half_kernel_size = (gs_kernel_size - 1) // 2
sample_times = 120
img = np.array(Image.open(image_path))

# taichi fields definition
gaussian_kernel_tf = ti.field(dtype=ti.f32, shape=(gs_kernel_size, gs_kernel_size))
img_tf = ti.field(dtype=ti.u8, shape=img.shape)
temp_u32_tf = ti.field(dtype=ti.u32, shape=img.shape)
temp_u32_2_tf = ti.field(dtype=ti.u32, shape=img.shape)
temp_f32_tf = ti.field(dtype=ti.f32, shape=img.shape)
out_img_tf = ti.field(dtype=ti.u8, shape=img.shape)

def generate_gaussian_kernel(size, sigma):
    # 创建一个空的高斯核
    kernel = np.zeros(size)
    
    # 计算中心点的坐标
    center_x = size[0] // 2
    center_y = size[1] // 2
    
    # 生成高斯核
    for x in range(size[0]):  # 遍历行
        for y in range(size[1]):  # 遍历列
            # 计算当前位置到中心点的距离
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2)
            # 计算高斯值并将其赋给当前位置
            kernel[x, y] = np.exp(-distance)
    
    # 将中心元素设置为最大值
    kernel[center_x, center_y] = 1
    
    # 对每个通道的核进行归一化确保和为1
    kernel[:, :] /= np.sum(kernel[:, :])
    
    return kernel

def load_data():
    img_tf.from_numpy(img)
    gaussian_kernel_tf.from_numpy(generate_gaussian_kernel(size=(gs_kernel_size, gs_kernel_size), sigma=sigma))

# 0.14s
@ti.kernel
def average_filter_impl(max_h: int, max_v: int, max_c: int):
    # every region that need to be synchronized should be written in a block (like for loop)
    # vertical filter 1
    for h, v, c in ti.ndrange(max_h, max_v, max_c):
        for offset in range(-half_kernel_size, half_kernel_size):
            temp_u32_tf[h, v, c] += img_tf[(h + offset) % max_h, v, c]
        temp_u32_tf[h, v, c] //= 2 * half_kernel_size
        pass
    # horizontal filter 1
    for h, v, c in ti.ndrange(max_h, max_v, max_c):
        for offset in range(-half_kernel_size, half_kernel_size):
            temp_u32_2_tf[h, v, c] += temp_u32_tf[h, (v + offset) % max_v, c]
        temp_u32_2_tf[h, v, c] //= 2 * half_kernel_size
        pass

    # vertical filter 2
    for h, v, c in ti.ndrange(max_h, max_v, max_c):
        # reset temp_tf to 0.
        temp_u32_tf[h, v, c] = 0
        for offset in range(-half_kernel_size, half_kernel_size):
            temp_u32_tf[h, v, c] += temp_u32_2_tf[(h + offset) % max_h, v, c]
        temp_u32_tf[h, v, c] //= 2 * half_kernel_size
        pass
    
    # horizontal filter 2
    for h, v, c in ti.ndrange(max_h, max_v, max_c):
        # reset temp_2_tf to 0.
        temp_u32_2_tf[h, v, c] = 0
        for offset in range(-half_kernel_size, half_kernel_size):
            temp_u32_2_tf[h, v, c] += temp_u32_tf[h, (v + offset) % max_v, c]
        temp_u32_2_tf[h, v, c] //= 2 * half_kernel_size
        out_img_tf[h, v, c] = temp_u32_2_tf[h, v, c]
        pass
    pass


@ti.kernel
def gaussian_filter_impl(max_h: int, max_v: int, max_c: int):
    for h, v, c in ti.ndrange(max_h, max_v, max_c):
        for offset_h, offset_v in ti.ndrange((-gs_half_kernel_size, gs_half_kernel_size + 1), (-gs_half_kernel_size, gs_half_kernel_size + 1)):
            temp_f32_tf[h, v, c] += img_tf[(h + offset_h) % max_h, (v + offset_v) % max_v, c] * gaussian_kernel_tf[gs_half_kernel_size + offset_h, gs_half_kernel_size + offset_v]
            pass
        out_img_tf[h, v, c] = ti.u8(temp_f32_tf[h, v, c])
        pass
    pass

@ti.kernel
def gaussian_sample_filter_impl(max_h: int, max_v: int, max_c: int):
    for h, v, c in ti.ndrange(max_h, max_v, max_c):
        for i in range(sample_times):
            offset_h = int(ti.random() * 2 * gs_half_kernel_size + 1 - gs_half_kernel_size)
            offset_v = int(ti.random() * 2 * gs_half_kernel_size + 1 - gs_half_kernel_size)
            temp_f32_tf[h, v, c] += img_tf[(h + offset_h) % max_h, (v + offset_v) % max_v, c] * gaussian_kernel_tf[gs_half_kernel_size + offset_h, gs_half_kernel_size + offset_v]
            pass
        out_img_tf[h, v, c] = ti.u8(min(255, temp_f32_tf[h, v, c] * (2 * gs_half_kernel_size) * (2 * gs_half_kernel_size) / sample_times))
        pass
    pass

def elapsed(func, *args):
    t1 = time.time()
    func(*args)
    ti.sync()
    return time.time() - t1

def test():
    # print(temp_f32_tf)
    # print(gaussian_kernel_tf)
    pass


def show():
    gui = ti.GUI(res=(img.shape[0], img.shape[1]))
    while gui.running:
        gui.set_image(img=out_img_tf)
        gui.show()
    pass

def filter():
    max_h, max_v, max_c = img.shape
    # print(f"Elapsed time: {elapsed(average_filter_impl, max_h, max_v, max_c)}") # 0.14s
    # print(f"Elapsed time: {elapsed(gaussian_filter_impl, max_h, max_v, max_c)}") # 0.27s
    print(f"Elapsed time: {elapsed(gaussian_sample_filter_impl, max_h, max_v, max_c)}") # 0.09s
    pass

def main():
    print(img.shape)
    load_data()
    filter()
    test()
    show()
    pass

if __name__ == "__main__":
    main()
    pass