from PIL import Image
import numpy as np
import taichi as ti
from pathlib import Path
import time
ti.init(arch=ti.gpu)

image_path = "ImageGradientBlur/123.jpg"
out_folder = "out"
half_kernel_size = 30
img = np.array(Image.open(image_path))

# taichi fields definition
img_tf = ti.field(dtype=ti.u8, shape=img.shape)
temp_tf = ti.field(dtype=ti.u32, shape=img.shape)
temp_2_tf = ti.field(dtype=ti.u32, shape=img.shape)
out_img_tf = ti.field(dtype=ti.u8, shape=img.shape)

def load_image():
    img_tf.from_numpy(img)
    


@ti.kernel
def average_filter_impl(max_h: int, max_v: int, max_c: int):
    # every region that need to be synchronized should be written in a block (like for loop)
    # init field
    for h, v, c in ti.ndrange(max_h, max_v, max_c):
        temp_tf[h, v, c] = 0
        temp_2_tf[h, v, c] = 0
        out_img_tf[h, v, c] = 0
        pass
    # vertical filter 1
    for h, v, c in ti.ndrange(max_h, max_v, max_c):
        for offset in range(-half_kernel_size, half_kernel_size):
            temp_tf[h, v, c] += img_tf[(h + offset) % max_h, v, c]
        temp_tf[h, v, c] //= 2 * half_kernel_size
        pass
    # horizontal filter 1
    for h, v, c in ti.ndrange(max_h, max_v, max_c):
        for offset in range(-half_kernel_size, half_kernel_size):
            temp_2_tf[h, v, c] += temp_tf[h, (v + offset) % max_v, c]
        temp_2_tf[h, v, c] //= 2 * half_kernel_size
        pass

    # vertical filter 2
    for h, v, c in ti.ndrange(max_h, max_v, max_c):
        # reset temp_tf to 0.
        temp_tf[h, v, c] = 0
        for offset in range(-half_kernel_size, half_kernel_size):
            temp_tf[h, v, c] += temp_2_tf[(h + offset) % max_h, v, c]
        temp_tf[h, v, c] //= 2 * half_kernel_size
        pass
    
    # horizontal filter 2
    for h, v, c in ti.ndrange(max_h, max_v, max_c):
        # reset temp_2_tf to 0.
        temp_2_tf[h, v, c] = 0
        for offset in range(-half_kernel_size, half_kernel_size):
            temp_2_tf[h, v, c] += temp_tf[h, (v + offset) % max_v, c]
        temp_2_tf[h, v, c] //= 2 * half_kernel_size
        out_img_tf[h, v, c] = temp_2_tf[h, v, c]
        pass
    pass

def elapsed(func, *args):
    t1 = time.time()
    func(*args)
    return time.time() - t1

def test():
    # print(out_img_tf)
    pass


def show():
    gui = ti.GUI(res=(img.shape[0], img.shape[1]))
    while gui.running:
        gui.set_image(img=out_img_tf)
        gui.show()
    pass

def average_filter():
    max_h, max_v, max_c = img.shape
    print(f"Elapsed time: {elapsed(average_filter_impl, max_h, max_v, max_c)}")
    ti.sync()
    pass

def main():
    print(img.shape)
    load_image()
    average_filter()
    test()
    show()
    pass

if __name__ == "__main__":
    main()
    pass