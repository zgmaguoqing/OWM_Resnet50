import sys, time
import numpy as np
import torch
import torch.nn as nn
from myresnet import owm_Conv2d

if torch.cuda.is_available():# run on GPU
    device = 'cuda'
else:
    device = 'cpu'
import utils


########################################################################################################################

class Appr(object):

    def __init__(self, model, nepochs=0, sbatch=64, lr=0,  clipgrad=10, args=None):
        self.model = model

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.clipgrad = clipgrad

        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()




        self.test_max = 0

        return

    def _get_optimizer(self, t=0, lr=None):
        # if lr is None:
        #     lr = self.lr
        lr = self.lr
        lr_owm = self.lr
        # fc1_params = list(map(id, self.model.fc1.parameters()))
        # fc2_params = list(map(id, self.model.fc2.parameters()))
        # fc3_params = list(map(id, self.model.fc3.parameters()))
        # base_params = filter(lambda p: id(p) not in fc1_params + fc2_params + fc3_params,
        #                      self.model.parameters())
        # optimizer = torch.optim.SGD([{'params': base_params},
        #                              {'params': self.model.fc1.parameters(), 'lr': lr_owm},
        #                              {'params': self.model.fc2.parameters(), 'lr': lr_owm},
        #                              {'params': self.model.fc3.parameters(), 'lr': lr_owm}
        #                              ], lr=lr, momentum=0.9)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

        return optimizer

    def train(self, t, xtrain, ytrain, xvalid, yvalid, data, ifowm = True, t_num = 2):
        num_task = 10 // t_num
        best_loss = np.inf
        best_acc = 0
        best_model = utils.get_model(self.model)
        lr = self.lr
        # patience = self.lr_patience
        self.optimizer = self._get_optimizer(t, lr)
        nepochs = self.nepochs
        test_max = 0
        # Loop epochs
        try:
            for e in range(nepochs):
                # Train

                self.train_epoch(xtrain, ytrain, cur_epoch=e, nepoch=nepochs, ifowm = ifowm)
                train_loss, train_acc = self.eval(xtrain, ytrain)
                print('| [{:d}/{:d}], Epoch {:d}/{:d}, | Train: loss={:.3f}, acc={:2.2f}% |'.format(t + 1, num_task, e + 1,
                                                                                                 nepochs, train_loss,
                                                                                                 100 * train_acc),
                      end='')
                # # Valid
                valid_loss, valid_acc = self.eval(xvalid, yvalid)
                print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss, 100 * valid_acc), end='')
                print()

                
                xtest = data[num_task]['test']['x'].cuda()
                ytest = data[num_task]['test']['y'].cuda()

                # xtest = data[1]['test']['x'].cuda()
                # ytest = data[1]['test']['y'].cuda()

                _, test_acc = self.eval(xtest, ytest)

                # # Adapt lr
                # if valid_loss < best_loss:
                #     best_loss = min(best_loss,valid_loss)

                # if valid_acc > best_acc:
                #     best_acc = max(best_acc, valid_acc)
                if test_acc>self.test_max:
                    self.test_max = max(self.test_max, test_acc)
                    best_model = utils.get_model(self.model)

                print('>>> Test on All Task:->>> Max_acc : {:2.2f}%  Curr_acc : {:2.2f}%<<<'.format(100 * self.test_max, 100 * test_acc))

        except KeyboardInterrupt:
            print()

        # Restore best validation model
        utils.set_model_(self.model, best_model)
        return

    def train_epoch(self, x, y, cur_epoch=0, nepoch=0, ifowm = True):
        self.model.train()

        r_len = np.arange(x.size(0))
        np.random.shuffle(r_len)
        r_len = torch.LongTensor(r_len).cuda()

        # Loop batches
        for i_batch in range(0, len(r_len), self.sbatch):
            b = r_len[i_batch:min(i_batch + self.sbatch, len(r_len))]
            images = x[b].clone()
            targets = y[b].clone()
            # print(images.max())
            # raise ValueError("Dev mode")

            # Forward
            output, h_list, x_list = self.model.forward(images)
            loss = self.ce(output, targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            lamda = i_batch / len(r_len)/nepoch + cur_epoch/nepoch

            alpha_array = [1.0 * 0.00001 ** lamda, 1.0 * 0.0001 ** lamda, 1.0 * 0.01 ** lamda, 1.0 * 0.1 ** lamda]

            def pro_weight(p, x, w, alpha=1.0, cnn=True, stride=1):
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
                    else:
                        r = x
                        k = torch.mm(p, torch.t(r))
                        p.sub_(torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
                        w.grad.data = torch.mm(w.grad.data, torch.t(p.data))
            def pro_net(net,block):
                for layer in net.children():
                    if isinstance(layer,owm_Conv2d):
                        layer.pro_weight(layer.P,layer.x,layer.weight)
                    elif isinstance(layer,nn.Linear):
                        layer.pro_weight(layer.P, layer.x, layer.weight,layer.bias)
                    elif isinstance(layer,block) or isinstance(layer,nn.Sequential):
                        pro_net(layer,block)

            # Compensate embedding gradients
            if ifowm:
                pro_net(self.model,self.model.block)

            # Apply step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
        return

    def eval(self, x, y):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0, len(r), self.sbatch):
            b = r[i:min(i + self.sbatch, len(r))]
            images = x[b].clone()
            targets = y[b].clone()

            # Forward
            output,  _, _ = self.model.forward(images)
            loss = self.ce(output, targets)
            _, pred = output.max(1)
            hits = (pred % 10 == targets).float()

            # Log
            total_loss += loss.data.cpu().numpy().item() * len(b)
            total_acc += hits.sum().data.cpu().numpy().item()
            total_num += len(b)

        return total_loss / total_num, total_acc / total_num
