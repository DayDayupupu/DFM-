import cv2
import os


def save_edges(input_folder, output_folder):
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有图片
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # 只处理图像文件
            # 读取图像
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"无法读取图像: {image_path}")
                continue

            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 使用 Canny 边缘检测
            edges = cv2.Canny(gray, 100, 200)

            # 保存边缘图像
            edge_filename = os.path.join(output_folder, f"edge_{filename}")
            cv2.imwrite(edge_filename, edges)

            print(f"保存边缘图像: {edge_filename}")


# 示例使用
input_folder = 'E:/guobiao/DFM-Net-Extension/data/RGBD_for_train/RGB'  # 替换为你的输入文件夹路径
output_folder = 'E:/guobiao/DFM-Net-Extension/data/RGBD_for_train/edge'  # 替换为你的输出文件夹路径

save_edges(input_folder, output_folder)
