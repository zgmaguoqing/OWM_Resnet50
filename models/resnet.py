import torch 
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class IncreResNet(ResNet):
    def __init__(self, block, layers):
        super().__init__(block=block, layers=layers)
        self.out_features = self.fc.in_features
        del self.fc
        
    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x 
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.logits(x)
        return x

    def logits(self, x):
        x = self.last(x)
        return x
    
def resnet18():
    model = IncreResNet(BasicBlock, [2, 2, 2, 2])
    model.conv1 = conv3x3(3, 64)
    return model

def resnet50():
    return IncreResNet(Bottleneck, [3, 4, 6, 3],)