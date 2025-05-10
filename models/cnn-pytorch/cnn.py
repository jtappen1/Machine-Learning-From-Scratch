import torch.nn as nn

class BasicCnn(nn.Module):
    def __init__(self, num_classes=10):
        super(BasicCnn, self).__init__()
        # Start with an input of 32x32
        # if we start at the first index 0,0 we will compute 32 filters over the image.  Going for a deeper representation
        # so I am also going to do 2 conv layers 
        self.conv_layer_1 = nn.Conv2d(in_channels=3,out_channels=32, kernel_size=3, stride =1)
        self.conv_layer_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)

        # we want to reduce the number of features, so we are going to pool these.
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(1600, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)