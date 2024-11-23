from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor
# 输入目录
input_dir = 'D:/baseline2/data/val/DATA_clean'


# 定义旋转图像并保存的函数
def rotate_and_save_image(filename):
    input_path = os.path.join(input_dir, filename)

    try:
        # 打开图像
        img = Image.open(input_path)

        # 旋转图像90度
        img_rotated = img.transpose(Image.ROTATE_270)   # 顺时针旋转90度

        # 构建新的文件名，原文件名 + "_90L"
        output_filename = f"{os.path.splitext(filename)[0]}_90L.png"
        output_path = os.path.join(input_dir, output_filename)

        # 保存旋转后的图像到原文件夹
        img_rotated.save(output_path)
        print(f"Processed {filename} -> {output_filename}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")


# 获取目录中的所有png文件
filenames = [f for f in os.listdir(input_dir) if f.endswith('.png')]

# 使用ThreadPoolExecutor并行处理图像
with ThreadPoolExecutor() as executor:
    executor.map(rotate_and_save_image, filenames)
