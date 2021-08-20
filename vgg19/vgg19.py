"""
Model adapted to extract feature activations from the PyTorch source code
for VGG models (https://pytorch.org/vision/stable/_modules/torchvision
/models/vgg.html)
"""

import torch.nn as nn


class VGG19(nn.Module):

    def __init__(self, num_classes=1000):
        super(VGG19, self).__init__()

        self.layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.ReLU(inplace=True)
        ),
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1,
                             ceil_mode=False)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1,
                             ceil_mode=False)
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1,
                             ceil_mode=False)
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1,
                             ceil_mode=False)
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1,
                             ceil_mode=False)
            ),
            nn.Sequential(
                nn.Linear(in_features=512 * 7 * 7, out_features=4096,
                          bias=True),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=4096, out_features=4096, bias=True),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=4096, out_features=num_classes,
                          bias=True)
            )])

    def forward(self, x):
        activations = []
        for i, layer in enumerate(self.layers):
            if i == 0:
                activations.append(layer(x))
            elif i == 16:
                out15_reshaped = activations[i-1].view \
                    (activations[i-1].size(0), 512 * 7 * 7)
                activations.append(layer(out15_reshaped))
            else:
                activations.append(layer(activations[i - 1]))

        return activations
