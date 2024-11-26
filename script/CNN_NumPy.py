import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.functional as F
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define RGB convolve function.
def convolve(image, kernel):
    # convert to PyTorch Tensor (C, H, W)
    image_tensor = torch.tensor(image, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
    kernel_tensor = torch.tensor(kernel, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    # convolve to each channel
    result_channels = []
    for c in range(image_tensor.shape[1]):
        channel = image_tensor[:, c:c+1, :, :]
        result_channel = F.conv2d(channel, kernel_tensor, padding=kernel_tensor.shape[-1] // 2)  # 卷积操作
        result_channels.append(result_channel)

    # merge all channel
    result = torch.cat(result_channels, dim=1).squeeze(0).permute(1, 2, 0)
    return result.cpu().numpy()

print("Start...")

# Read image
image = cv2.imread("resources/RaindowStudioHomePageLightMode.jpg")  # read by BGR
print("Image loaded start convert to RGB...")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB
print("Image converted...")

# Define kernels
filters = [
    ("Original", None),
    ("Sharpened", np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])),
    ("Diagonal Edged", np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])),
    ("Gaussian Blurred", np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16),
    ("Blurred", np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9),
    ("Edged", np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]))
]

# Apply kernels
processed_images = [image]

for title, kernel in tqdm(filters[1:], desc="Applying Filters", unit="filter"):
    start_time = time.time()
    processed_image = convolve(image, kernel)
    elapsed_time = time.time() - start_time
    print(f"{title}: Completed in {elapsed_time:.2f} seconds")
    processed_images.append(processed_image)

# Show Result

for i, (title, img) in enumerate(zip([f[0] for f in filters], processed_images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(np.clip(img, 0, 255).astype(np.uint8))
    plt.title(title)
    plt.axis('off')
    
plt.tight_layout()
plt.show()