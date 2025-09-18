import os
import torch
import torch.nn as nn
from types import MethodType
import models
from utils.metric import accumulate_acc, get_pred, AverageMeter, Timer
import torch.nn.utils.weight_norm as WeightNorm
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.dist_utils import distributed_concat, get_world_size, reduce_tensor_sum
import torch.distributed as dist


class CLTrainer(object):
    '''
    Normal Neural Network with SGD for classification
    '''

    def __init__(self, config, args, logger, out_dim, ckpt_path=None):
        super().__init__()
        self.config = {
            'model_type': config.AGENT.MODEL_TYPE, 'model_name': config.AGENT.MODEL_NAME,
            'model_weights': ckpt_path, 'out_dim': out_dim,
            'optimizer': config.OPT.NAME, 'lr': config.OPT.LR, 'momentum': config.OPT.MOMENTUM,
            'weight_decay': config.OPT.WEIGHT_DECAY, 'schedule': config.OPT.SCHEDULE, 'gamma': config.OPT.GAMMA,
            'print_freq': config.PRINT_FREQ, 'gpuid': config.GPUID,
            'reg_coef': config.AGENT.REG_COEF,
        }
        self.args = args
        self.log = logger
        self.task_count = args.task_count
        self.num_task = config.DATASET.NUM_TASKS

        # If out_dim is a dict, there is a list of tasks. The model will have a head for each task.
        self.multihead = True if len(
            self.config['out_dim']) > 1 else False  # A convenience flag to indicate multi-head/task
        model = self.create_model()
        criterion_fn = nn.CrossEntropyLoss()
        self.set_device(model, criterion_fn)
        self.init_optimizer()
        self.reset_optimizer = True
        self.storage = None
        self.valid_out_dim = 'ALL'  # Default: 'ALL' means all output nodes are active
        # Set a interger here for the incremental class scenario

    def init_optimizer(self):
        optimizer_arg = {'params': self.model.parameters(),
                         'lr': self.config['lr'],
                         'weight_decay': self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD', 'RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'

        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['schedule'],
                                                              gamma=self.config['gamma'])

    def create_model(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']]()

        # Apply network surgery to the backbone
        # Create the heads for tasks (It can be single task or multi-task)
        n_feat = model.out_features

        # The output of the model will be a dict: {task_name1:output1, task_name2:output2 ...}
        # For a single-headed model the output will be {'All':output}
        model.last = nn.ModuleDict()
        for task, out_dim in cfg['out_dim'].items():
            if task == 'All':
                model.last[task] = WeightNorm(nn.Linear(n_feat, out_dim, bias=False))
            else:
                model.last[task] = nn.Linear(n_feat, out_dim, bias=False)

        # Redefine the task-dependent function
        def new_logits(self, x):
            outputs = {}
            for task, func in self.last.items():
                outputs[task] = func(x)
            return outputs

        # Replace the task-dependent function
        model.logits = MethodType(new_logits, model)
        # Load pre-trained weights
        if cfg['model_weights'] is not None:
            # if self.args.is_main_process:
            self.log.info('=> Load model weights: {}'.format(cfg['model_weights']))
            model_state = torch.load(cfg['model_weights'],
                                     map_location=lambda storage, loc: storage)  # Load to CPU.
            model.load_state_dict(model_state)
            # if self.args.is_main_process:
            self.log.info('=> Load Done')
        return model

    def forward(self, x):
        return self.model.forward(x)

    def predict(self, inputs):
        self.model.eval()
        out = self.forward(inputs)
        for t in out.keys():
            out[t] = out[t].detach()
        return out

    def validation(self, dataloader):
        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = self.model.training
        self.model.eval()
        with torch.no_grad():
            preds = []
            targets = []
            for i, (input, target, task) in enumerate(dataloader):
                with torch.no_grad():
                    input = input.to(self.device)
                    target = target.to(self.device)
                output = self.predict(input)

                # Summarize the performance of all tasks, or 1 task, depends on dataloader.
                # Calculated by total number of data.

                # In case that the task id is unknown, ``task" can only be used to compute
                # the final score but not for inference.
                if self.args.distributed:
                    pred = get_pred(output, task)
                    preds.append(pred)
                    targets.append(target)
                else:
                    acc = accumulate_acc(output, target, task, acc)
            if self.args.distributed:
                # all data
                dsize = len(dataloader.dataset)
                preds = distributed_concat(torch.concat(preds, dim=0), dsize)
                targets = distributed_concat(torch.concat(targets, dim=0), dsize)
                acc.update((preds == targets).view(-1).sum().float().item() / dsize * 100, 1)

        self.model.train(orig_mode)
        # if self.args.is_main_process:
        self.log.info(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                      .format(acc=acc, time=batch_timer.toc()))
        return acc.avg

    def test(self, dataloader):
        orig_mode = self.model.training
        self.model.eval()
        res = []
        for i, (inputs, task) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                output = self.predict(inputs)
                pred = get_pred(output, task)
            res.append(pred)
        if self.args.distributed:
            dsize = len(dataloader.dataset)
            res = distributed_concat(torch.concat(res, dim=0), dsize)
        else:
            res = torch.cat(res, dim=0)

        self.model.train(orig_mode)
        return res

    def criterion(self, preds, targets, tasks, **kwargs):
        # The inputs and targets could come from single task or a mix of tasks
        # The network always makes the predictions with all its heads
        # The criterion will match the head and task to calculate the loss.
        if self.multihead:
            loss = 0
            for t, t_preds in preds.items():
                inds = [i for i in range(len(tasks)) if tasks[i] == t]  # The index of inputs that matched specific task
                if len(inds) > 0:
                    t_preds = t_preds[inds]
                    t_target = targets[inds]
                    loss += self.criterion_fn(t_preds, t_target) * len(inds)  # restore the loss from average
            loss /= len(targets)  # Average the total loss by the mini-batch size
        else:
            pred = preds['All']
            if isinstance(self.valid_out_dim,
                          int):  # (Not 'ALL') Mask out the outputs of unseen classes for incremental class scenario
                pred = preds['All'][:, :self.valid_out_dim]
            loss = self.criterion_fn(pred, targets)
        return loss

    def update_model(self, inputs, targets, tasks):
        out = self.forward(inputs)
        loss = self.criterion(out, targets, tasks)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach(), out

    def learn_tasks(self, train_loader, val_loader=None):
        self.before_tasks(train_loader)
        self.train_model(train_loader, val_loader)
        self.after_tasks(train_loader)

    def before_tasks(self, *args, **kwargs):
        if self.reset_optimizer:  # Reset optimizer before learning each task
            # if self.args.is_main_process:
            self.log.info('Optimizer is reset!')
            self.init_optimizer()

    def after_tasks(self, *args, **kwargs):
        pass

    def before_epoch(self, *args, **kwargs):
        pass

    def after_epoch(self, *args, **kwargs):
        pass

    def train_model(self, train_loader, val_loader):
        for epoch in range(self.config['schedule'][-1]):
            # Config the model and optimizer
            # if self.args.is_main_process:
            self.log.info('Epoch:{0}'.format(epoch))
            if self.args.distributed:
                train_loader.sampler.set_epoch(epoch)
            self.model.train()
            self.scheduler.step(epoch)
            # if self.args.is_main_process:
            for param_group in self.optimizer.param_groups:
                self.log.info('LR:{}'.format(param_group['lr']))

            if self.args.is_main_process:
                log_str = ' Itr\t    Time  \t    Data  \t  Loss  \t  Acc'
            if self.args.distributed:
                log_str = 'Rank\t' + log_str
            self.log.info(log_str)

            self.before_epoch()
            self.train_epoch(train_loader)

            if self.args.distributed:
                dist.barrier()

            self.after_epoch()
            # Evaluate the performance of current task
            if val_loader != None:
                self.validation(val_loader)

    def train_epoch(self, train_loader):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        data_timer = Timer()
        batch_timer = Timer()
        data_timer.tic()
        batch_timer.tic()

        for i, (input, target, task) in enumerate(train_loader):
            data_time.update(data_timer.toc())  # measure data loading time
            input = input.to(self.device)
            target = target.to(self.device)

            loss, output = self.update_model(input, target, task)
            input = input.detach()
            target = target.detach()

            # measure accuracy and record loss
            acc = accumulate_acc(output, target, task, acc, reduce=False)
            # if get_world_size() > 1:
            #     loss = reduce_tensor_sum(loss.detach()) / get_world_size()
            losses.update(loss, input.size(0))

            batch_time.update(batch_timer.toc())  # measure elapsed time
            data_timer.toc()
            # if self.args.is_main_process:
            if ((self.config['print_freq'] > 0) and (i % self.config['print_freq'] == 0)) or (i + 1) == len(
                    train_loader):
                log_str = '[{0}/{1}]\t'.format(i, len(train_loader))
                log_str += '{batch_time.val:.4f}({batch_time.avg:.4f})\t'.format(batch_time=batch_time)
                log_str += '{data_time.val:.4f}({data_time.avg:.4f})\t'.format(data_time=data_time)
                log_str += '{loss.val:.3f}({loss.avg:.3f})\t'.format(loss=losses)
                log_str += '{acc.val:.2f}({acc.avg:.2f})'.format(acc=acc)
                if self.args.distributed:
                    log_str = f'  {self.args.local_rank}\t' + log_str
                self.log.info(log_str)

        self.log.info(' * Train Acc {acc.avg:.3f}'.format(acc=acc))
        return losses.avg, acc.avg

    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())

    def save_model(self, filename):
        model_state = self.model.state_dict()
        if isinstance(self.model, DDP):
            # Get rid of 'module' before the name of states
            model_state = self.model.module.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        if self.args.is_main_process:
            print('=> Saving model to:', filename)
        torch.save(model_state, filename)
        if self.args.is_main_process:
            print('=> Save Done')

    def set_device(self, model, criterion_fn):
        if self.args.distributed:
            self.device = torch.device("cuda", self.args.local_rank)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.device)
            self.model = DDP(
                model, device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True
            )
            self.criterion_fn = criterion_fn.to(self.device)
        else:
            if self.config['gpuid'][0] >= 0:
                torch.cuda.set_device(self.config['gpuid'][0])
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
            self.model = model.to(self.device)
            self.criterion_fn = criterion_fn.to(self.device)

    def load_storage(self, path):
        self.log.info('load storage...')
        storages = torch.load(path)
        memory_ = os.path.getsize(path) / float(1024 * 1024)
        self.allocate_memory(storages['storage'])
        self.log.info('done')
        return memory_

    def save_storage(self, path):
        self.log.info('saving storage...')
        storage = self.collect_memoey()
        storages = {
            'storage': storage,
        }
        torch.save(storages, path)
        self.log.info('done')

    def collect_memoey(self) -> dict:
        '''
        Return a variable of dictionary type,
        each value in the dictionary should be of type ``float32".
        '''
        pass
        # raise NotImplementedError

    def allocate_memory(self, storage):
        pass
        # raise NotImplementedError



