from PIL import Image, ImageFilter

def blur_image(image_path):
    # 打开图片
    image = Image.open(image_path)
    
    # 应用平均模糊滤镜
    blurred_image = image.filter(ImageFilter.BLUR)
    
    # 保存模糊后的图片
    output_path = "blurred_image.png"
    blurred_image.save(output_path)
    print("图片模糊处理完成，已保存为", output_path)

# 使用示例
image_path = "123.jpg"
blur_image(image_path)
