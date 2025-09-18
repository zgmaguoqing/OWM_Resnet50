import os
import time
import torch
from utils.dist_utils import reduce_tensor_sum, get_world_size

def compute_acc_bwt(acc_table, num_tasks):
    avg_acc_history = [0] * num_tasks
    avg_bwt_history = [0] * num_tasks
    for i in range(num_tasks):
        train_name = str(i)
        cls_acc_sum = 0
        bwt = 0
        for j in range(i + 1):
            val_name = str(j)
            cls_acc_sum += acc_table[val_name][train_name]
            bwt += acc_table[val_name][train_name] - \
                acc_table[val_name][val_name]
        avg_acc_history[i] = cls_acc_sum / (i + 1)
        avg_bwt_history[i] = bwt / i if i > 0 else 0

    acc = avg_acc_history[-1]
    bwt = avg_bwt_history[-1]
    return acc, bwt

def accumulate_acc(output, target, task, meter, reduce=True):
    if 'All' in output.keys(): # Single-headed model
        acc = accuracy(output['All'], target)
        if get_world_size() > 1 and reduce:
            acc = reduce_tensor_sum(acc)  / get_world_size()
        meter.update(acc.item(), len(target))
    else:  # outputs from multi-headed (multi-task) model
        for t, t_out in output.items():
            inds = [i for i in range(len(task)) if task[i] == t]  # The index of inputs that matched specific task
            if len(inds) > 0:
                t_out = t_out[inds]
                t_target = target[inds]
                acc = accuracy(t_out, t_target)
                if get_world_size() > 1 and reduce:
                    acc = reduce_tensor_sum(acc) / get_world_size()
                meter.update(acc.item(), len(inds))
    return meter

def get_pred(output, task, topk=(1,)):
    if  'All' in output.keys():
        out = output['All']
    else:
        for t, t_out in output.items():
            inds = [i for i in range(len(task)) if task[i] == t]  # The index of inputs that matched specific task
            if len(inds) > 0:
                out = t_out[inds]
    maxk = max(topk)
    _, pred = out.topk(maxk, 1)
    return pred.squeeze(dim=1)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum()
            res.append(correct_k*100.0 / batch_size)

        if len(res)==1:
            return res[0]
        else:
            return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / self.count


class Timer(object):
    """
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.interval = 0
        self.time = time.time()

    def value(self):
        return time.time() - self.time

    def tic(self):
        self.time = time.time()

    def toc(self):
        self.interval = time.time() - self.time
        self.time = time.time()
        return self.interval