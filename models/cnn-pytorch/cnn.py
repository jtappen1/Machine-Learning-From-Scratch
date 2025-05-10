import torch.nn as nn

# https://docs.pytorch.org/docs/stable/tensors.html
class BasicCNN(nn.Module):

    def __init__(self, num_classes=10):
        super(BasicCNN, self).__init__()
        # Start with an input of 32x32
        # if we start at the first index 0,0 we will compute 32 filters over the image.  Going for a deeper representation
        # so I am also going to do 2 conv layers

        # Input: [batch_size = N, 3 (RGB), 32(H), 32(W)]
        # Output: For each spatial dimension: output = ([Input - kernel_size + 2 * padding] / stride) + 1
        self.conv_layer_1 = nn.Conv2d(in_channels=3,out_channels=32, kernel_size=3, stride =1, padding=1)
        self.conv_layer_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        # we want to reduce the number of features, so add a max pooling layer.
        # Accepts dimensionality of [N, 32, 32, 32], W x H x D = 32  x 32 x 32
        # F = kernel_size
        # S = stride
        # Reduces dimensionality by ((W - F)/S) + 1 -> 32-2/2 + 1 = 16
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: [N, 32, 16, 16]

        # Input: [N, 32, 16,16]
        # Preserves size with padding, increase depth to 64
        self.conv_layer_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Output: [N, 64, 16, 16]
        self.conv_layer_4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # Output: [N, 64, 16, 16]

        # One thing to remember: Max Pool does not effect the depth/channels.  Only the size of the features.
        # Input: [N, 64, 16, 16]
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: [N, 64, 8, 8]
        
        # Flatten for the Fully Connected Layer 8 x 8 x 64 = 4096
        self.fc1 = nn.Linear(4096, 128)
        # Output: [N, 128]
        self.relu1 = nn.ReLU()
        # Output: [N, 128]
        self.fc2 = nn.Linear(128, num_classes)
        # Output: [N, num_classes]
    
    def forward(self, x):
        # Compute the forward pass 
        # First block
        out = self.conv_layer_1(x)
        out = self.conv_layer_2(out)
        out = self.max_pool_1(out)

        # Second block
        out = self.conv_layer_3(out)
        out = self.conv_layer_4(out)
        out = self.max_pool_2(out)
        
        # Reshape for FC layer
        out = out.reshape(out.size(0), -1)

        # Output: [N, 4096]
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

