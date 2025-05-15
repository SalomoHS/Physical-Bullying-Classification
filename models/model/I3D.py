import torch
import torch.nn as nn

class Inception3D(nn.Module):
    def __init__(self, num_classes=6, dropout_prob=0.5):
        super(Inception3D, self).__init__()

        self.conv1 = nn.Conv3d(2, 64, kernel_size=(7,7,7), stride=(2,2,2), padding=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 192, kernel_size=(1,1,1), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv2b = nn.Conv3d(192, 192, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2))

        self.inception3a = self._make_inception_module(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = self._make_inception_module(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(1,3,1), stride=(1,2,1))

        self.inception3c = self._make_inception_module(480, 192, 96, 208, 16, 48, 64)
        self.inception4a = self._make_inception_module(512, 160, 112, 224, 24, 64, 64)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

        self.inception5a = self._make_inception_module(512, 128, 128, 256, 24, 64, 64)
        self.inception5b = self._make_inception_module(512, 112, 144, 288, 32, 64, 64)
        self.avgpool = nn.AdaptiveAvgPool3d((2,7,7))
        self.dropout = nn.Dropout3d(p=dropout_prob)

        self.conv6 = nn.Conv3d(528, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(1, 1, 1))

        self.fc = nn.Linear(1944, num_classes)  
        self.relu = nn.ReLU()

    def _make_inception_module(self, in_channels, ch1x1, ch3x3reduce, ch3x3, ch5x5reduce, ch5x5, pool_proj):
        return nn.ModuleDict({
            'branch1': nn.Conv3d(in_channels, ch1x1, kernel_size=(1, 1, 1)),
            'branch2': nn.Sequential(
                nn.Conv3d(in_channels, ch3x3reduce, kernel_size=(1, 1, 1)),
                nn.Conv3d(ch3x3reduce, ch3x3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            ),
            'branch3': nn.Sequential(
                nn.Conv3d(in_channels, ch5x5reduce, kernel_size=(1, 1, 1)),
                nn.Conv3d(ch5x5reduce, ch5x5, kernel_size=(3, 3, 3), padding=(1,1,1))
            ),
            'branch4': nn.Sequential(
                nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.Conv3d(in_channels, pool_proj, kernel_size=(1, 1, 1))
            )
        })

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv2b(x)
        x = self.maxpool2(x)

        x = self._forward_inception(self.inception3a, x)
        x = self._forward_inception(self.inception3b, x)
        x = self.maxpool3(x)

        x = self._forward_inception(self.inception3c, x)
        x = self._forward_inception(self.inception4a, x)
        x = self.maxpool4(x)

        x = self._forward_inception(self.inception5a, x)
        x = self._forward_inception(self.inception5b, x)
        x = self.avgpool(x)
        x = self.dropout(x)

        x =self.conv6(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.relu(x)
        return x

    def _forward_inception(self, inception_module, x):
        branch1 = inception_module['branch1'](x)
        branch2 = inception_module['branch2'](x)
        branch3 = inception_module['branch3'](x)
        branch4 = inception_module['branch4'](x)

        return torch.cat([branch1, branch2, branch3, branch4], 1)
