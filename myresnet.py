import torch.nn as nn
import math
import torch
from torch import Tensor
import torch.utils.model_zoo as model_zoo


# __all__ = ['myResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152']
#
#
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }
if torch.cuda.is_available():# run on GPU
    device = 'cuda'
else:
    device = 'cpu'
import utils

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class owm_Conv2d(nn.Conv2d):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,alpha=1.0,**kwargs):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         **kwargs)
        self.P=torch.eye(in_channels*kernel_size*kernel_size).to(device)
        self.x=None
        self.stride=stride
        self.alpha=alpha

    def forward(self, input: Tensor) -> Tensor:
        self.x=torch.mean(input, 0, True)
        output=super().forward(input)
        return output

    def pro_weight(self,p, x, w, alpha=1.0, cnn=True, stride=1):
        with torch.no_grad():
            if cnn:
                _, _, H, W = x.shape
                F, _, HH, WW = w.shape
                S = stride  # stride
                Ho = int(1 + (H - HH) / S)
                Wo = int(1 + (W - WW) / S)
                for i in range(Ho):
                    for j in range(Wo):
                        # N*C*HH*WW, C*HH*WW = N*C*HH*WW, sum -> N*1
                        r = x[:, :, i * S: i * S + HH, j * S: j * S + WW].contiguous().view(1, -1)
                        # r = r[:, range(r.shape[1] - 1, -1, -1)]
                        k = torch.mm(p, torch.t(r))
                        p.sub_(torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
                w.grad.data = torch.mm(w.grad.data.view(F, -1), torch.t(p.data)).view_as(w)

class owm_MLP(nn.Linear):
    def __init__(self,in_features,*args,bias=True,alpha=1.0,**kwargs):
        super().__init__(in_features,
                         *args,
                         bias=bias,
                         **kwargs)
        if bias:
            self.P=torch.eye(in_features+1).to(device)
        else:
            self.P = torch.eye(in_features).to(device)
        self.x=None
        self.bias_bool=bias
        self.alpha=alpha

    def forward(self, input: Tensor) -> Tensor:
        self.x=torch.mean(input, 0, True)
        output=super().forward(input)
        return output

    def pro_weight(self,p, x, w,b=None, alpha=1.0,):
        with torch.no_grad():
            if self.bias_bool:
                x=torch.cat([x,torch.tensor([[1]]).to(device)],dim=1)
                r = x
                k = torch.mm(p, torch.t(r))
                p.sub_(torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
                W_grad=torch.cat([w.grad.data, b.grad.data.unsqueeze(dim=1)], dim=1)
                Pro_W_grad=torch.mm(W_grad, torch.t(p.data))
                w.grad.data = Pro_W_grad[:,:-1]
                b.grad.data =Pro_W_grad[:,-1]

            else:
                r = x
                k = torch.mm(p, torch.t(r))
                p.sub_(torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
                w.grad.data = torch.mm(w.grad.data, torch.t(p.data))

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = owm_Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = owm_Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = owm_Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class myResNet(nn.Module):

    def __init__(self,inputsize, block, layers, num_classes=10):
        self.inplanes = 64
        super(myResNet, self).__init__()
        ncha, size, _ = inputsize
        self.block=block
        self.conv1 = owm_Conv2d(ncha, 64, kernel_size=5, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2) #3
        self.fc = owm_MLP(512 * block.expansion, num_classes)
        # self.sigmod = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1): # core neural layer setting:downsample,bottleneck
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                owm_Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        torch.autograd.set_detect_anomaly(True)
        h_list = []
        x_list = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        #### layer1
        x = self.layer1(x)


        #### layer2
        x = self.layer2(x)


        #### layer3
        x = self.layer3(x)

        #### layer4
        x = self.layer4(x)


        x = self.avgpool(x)
        x2 = x.view(x.size(0), -1)
        x = self.fc(x2)

        return x, h_list, x_list

    def resnet18():
        """Constructs a ResNet-18 model.
        """
        model = myResNet(BasicBlock, [2, 2, 2, 2])
        return model

    def resnet34():
        """Constructs a ResNet-34 model.
        """
        model = myResNet(BasicBlock, [3, 4, 6, 3])
        return model

    def resnet50(inputsize):
        """Constructs a ResNet-50 model.

        """
        model = myResNet(inputsize,Bottleneck, [3, 4, 6, 3])
        return model


def Net(inputsize):
    return myResNet.resnet50(inputsize)

if __name__=='__main__':
    # net=Net(12)
    print(1)