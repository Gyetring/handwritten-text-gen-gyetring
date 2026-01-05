from PIL import Image
import os
from skimage.morphology import skeletonize
import numpy as np
import cv2


def my_skeletonize(img, erode_kernel = (1,1)):
    # 转换为灰度图像
    img_gray = img.convert('L')  # 'L' 表示灰度模式
    # 调整大小
    img_resized = img_gray.resize((64, 64))
    
    img_np = np.array(img_resized)

    img_np = 255 - img_np

    # 1. 图像二值化 (去掉灰度值)
    _, binary_image = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY)

    # 2. 使用形态学腐蚀操作来去除细节
    kernel = np.ones(erode_kernel, np.uint8)  # 3x3的腐蚀核
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)

    # 3. 细化操作 (使用skimage的细化算法)
    # skimage的细化算法需要输入的是布尔类型的图像
    eroded_image = eroded_image // 255  # 转换为0和1
    skeleton = skeletonize(eroded_image)  # 细化

    # 4. 返回骨骼化后的图像（转回0-255范围）
    processed_image = (skeleton * 255).astype(np.uint8)

    processed_image = 255 - processed_image
    return processed_image

