import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class Net(torch.nn.Module):

    def __init__(self, inputsize,last_dim=4096,simclr_dim=32*4):
        super(Net, self).__init__()
        ncha, size, _ = inputsize
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.drop1 = torch.nn.Dropout(0.2)
        self.padding = torch.nn.ReplicationPad2d(1)

        self.c1 = torch.nn.Conv2d(ncha, 64, kernel_size=4, stride=1, padding=0, bias=False)
        self.c2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1,  padding=0, bias=False)
        self.c3 = torch.nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=0, bias=False)

        self.fc1 = torch.nn.Linear(256 * 4 * 4, 1024, bias=False)
        self.fc2 = torch.nn.Linear(1024, 1024, bias=False)
        self.fc3 = torch.nn.Linear(1024, 10,  bias=False)
        # self.out_num = 10
        # # self.weight3 = nn.Parameter(torch.Tensor(self.out_num, 300))
        # self.simclr_layer = nn.Sequential(
        #     nn.Linear(last_dim, last_dim),
        #     nn.ReLU(),
        #     nn.Linear(last_dim, simclr_dim),
        # )
        # self.shift_cls_layer = nn.Linear(last_dim, 4)

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.xavier_normal_(self.fc3.weight)

        return

    def conv_feature(self, x):
      #  h_list = []
        x_list = []
        import numpy as np
        num = np.random.choice(x.shape[0],1)
        # Gated
        x = self.padding(x)
        x_list.append(torch.mean(x, 0, True))
        #x_list.append(x[num[0]].unsqueeze(0))
        con1 = self.drop1(self.relu(self.c1(x)))
        con1_p = self.maxpool(con1)

        con1_p = self.padding(con1_p)
        x_list.append(torch.mean(con1_p, 0, True))
        #x_list.append(con1_p[num[0]].unsqueeze(0))
        con2 = self.drop1(self.relu(self.c2(con1_p)))
        con2_p = self.maxpool(con2)

        con2_p = self.padding(con2_p)
        x_list.append(torch.mean(con2_p, 0, True))

        #x_list.append(con2_p[num[0]].unsqueeze(0))
        con3 = self.drop1(self.relu(self.c3(con2_p)))
        con3_p = self.maxpool(con3)
        # print("b")
        h = con3_p.view(x.size(0), -1)

        return h,x_list,num[0]
    def forward(self, h,num,classify=True,shift=None,simclr=None):
        h_list = []


        h_list.append(torch.mean(h, 0, True))
        #h_list.append(h[num].unsqueeze(0))
        h = self.relu(self.fc1(h))

        h_list.append(torch.mean(h, 0, True))
        #h_list.append(h[num].unsqueeze(0))
        h = self.relu(self.fc2(h))

        h_list.append(torch.mean(h, 0, True))
        #h_list.append(h[num].unsqueeze(0))
        y = self.fc3(h)
        return y, h_list
        # else:
        #     _aux={}
        #     _return_aux = False

class Net100(torch.nn.Module):

    def __init__(self, inputsize,last_dim=4096,simclr_dim=32*4):
        super(Net100, self).__init__()
        ncha, size, _ = inputsize
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.drop1 = torch.nn.Dropout(0.2)
        self.padding = torch.nn.ReplicationPad2d(1)

        self.c1 = torch.nn.Conv2d(ncha, 64, kernel_size=4, stride=1, padding=0, bias=False)
        self.c2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1,  padding=0, bias=False)
        self.c3 = torch.nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=0, bias=False)

        self.fc1 = torch.nn.Linear(256 * 4 * 4, 1024, bias=False)
        self.fc2 = torch.nn.Linear(1024, 1024, bias=False)
        self.fc3 = torch.nn.Linear(1024, 100,  bias=False)
        # self.out_num = 10
        # # self.weight3 = nn.Parameter(torch.Tensor(self.out_num, 300))
        # self.simclr_layer = nn.Sequential(
        #     nn.Linear(last_dim, last_dim),
        #     nn.ReLU(),
        #     nn.Linear(last_dim, simclr_dim),
        # )
        # self.shift_cls_layer = nn.Linear(last_dim, 4)

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.xavier_normal_(self.fc3.weight)

        return

    def conv_feature(self, x):
      #  h_list = []
        x_list = []
        import numpy as np
        num = np.random.choice(x.shape[0],1)
        # Gated
        x = self.padding(x)
        x_list.append(torch.mean(x, 0, True))
        #x_list.append(x[num[0]].unsqueeze(0))
        con1 = self.drop1(self.relu(self.c1(x)))
        con1_p = self.maxpool(con1)

        con1_p = self.padding(con1_p)
        x_list.append(torch.mean(con1_p, 0, True))
        #x_list.append(con1_p[num[0]].unsqueeze(0))
        con2 = self.drop1(self.relu(self.c2(con1_p)))
        con2_p = self.maxpool(con2)

        con2_p = self.padding(con2_p)
        x_list.append(torch.mean(con2_p, 0, True))

        #x_list.append(con2_p[num[0]].unsqueeze(0))
        con3 = self.drop1(self.relu(self.c3(con2_p)))
        con3_p = self.maxpool(con3)
        # print("b")
        h = con3_p.view(x.size(0), -1)

        return h,x_list,num[0]
    def forward(self, h,num,classify=True,shift=None,simclr=None):
        h_list = []


        h_list.append(torch.mean(h, 0, True))
        #h_list.append(h[num].unsqueeze(0))
        h = self.relu(self.fc1(h))

        h_list.append(torch.mean(h, 0, True))
        #h_list.append(h[num].unsqueeze(0))
        h = self.relu(self.fc2(h))

        h_list.append(torch.mean(h, 0, True))
        #h_list.append(h[num].unsqueeze(0))
        y = self.fc3(h)
        return y, h_list
        # else:
        #     _aux={}
        #     _return_aux = False


class AlexNet(torch.nn.Module):

    def __init__(self, inputsize ):
        super(AlexNet, self).__init__()

        ncha, size, _ = inputsize
        #self.taskcla = taskcla

        self.conv1 = torch.nn.Conv2d(ncha, 64, kernel_size=size // 8)
        s = compute_conv_output_size(size, size // 8)
        s = s // 2
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=size // 10)
        s = compute_conv_output_size(s, size // 10)
        s = s // 2
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=2)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()

        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(256 * s * s, 2048)
        self.fc2 = torch.nn.Linear(2048, 2048)
        self.last = torch.nn.ModuleList()

        self.fc3 = torch.nn.Linear(2048, 10)

        return

    def forward(self, x):
        h_list = []
        x_list = []
        x_list.append(torch.mean(x, 0, True))
        h = self.maxpool(self.drop1(self.relu(self.conv1(x))))
        x_list.append(torch.mean(h, 0, True))
        h = self.maxpool(self.drop1(self.relu(self.conv2(h))))
        x_list.append(torch.mean(h, 0, True))
        h = self.maxpool(self.drop2(self.relu(self.conv3(h))))
        h = h.view(x.size(0), -1)
        h_list.append(torch.mean(h, 0, True))
        h = self.drop2(self.relu(self.fc1(h)))
        h_list.append(torch.mean(h, 0, True))
        h = self.drop2(self.relu(self.fc2(h)))
        h_list.append(torch.mean(h, 0, True))
        y = self.fc3(h)
        return y,h_list,x_list
