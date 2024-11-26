import torch.nn as nn

class simple(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), # keep image size 16*224*224
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16*112*112
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # keep image size 32*112*112
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # image size to 32*56*56
        )