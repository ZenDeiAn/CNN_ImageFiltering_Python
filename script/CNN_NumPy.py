import numpy as np
import cv2
from matplotlib import pyplot as plt

# 定义卷积函数
def convolve(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # 填充图像
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    result = np.zeros_like(image)
    
    # 卷积操作
    for i in range(h):
        for j in range(w):
            result[i, j] = np.sum(padded_image[i:i+kh, j:j+kw] * kernel)
    return result

# 读取图像
image = cv2.imread("resources/RaindowStudioLogo.png", cv2.IMREAD_GRAYSCALE)  # 读取灰度图

# 定义不同的卷积核
blur_kernel = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]]) / 9

sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

edge_kernel = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])

# 应用卷积滤波
blurred = convolve(image, blur_kernel)
sharpened = convolve(image, sharpen_kernel)
edges = convolve(image, edge_kernel)

# 显示结果
titles = ['Original', 'Blurred', 'Sharpened', 'Edges']
images = [image, blurred, sharpened, edges]

for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.show()