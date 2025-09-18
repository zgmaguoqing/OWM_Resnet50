import numpy as np
import torch
from agents.trainer import CLTrainer
import torch.distributed as dist

class OWMTrainer(CLTrainer):
    '''
    OWM CL strategy
    Train feature extractor on task-0, and freeze all parameters except for layer4.1.conv and layer4.2.conv. Learning them with OWM strategy on the left tasks.
    '''
    def __init__(self, config, args, logger, out_dim, ckpt_path=None):
        super().__init__(config, args, logger, out_dim, ckpt_path)
        self.storage_path = args.storage_path
        self.save_storage_path = args.save_storage_path
        self.ckpt_path = args.ckpt_path
        self.save_ckpt_path = args.save_ckpt_path
        self.task_type = config.DATASET.NAME # "10splitTasks", "4splitDomains"
        dtype = torch.cuda.FloatTensor  # run on GPU
        self.alpha0 = config.AGENT.ALPHA_ZERO
        self.alpha1 = config.AGENT.ALPHA_ONE
        self.alpha2 = config.AGENT.ALPHA_TWO
        self.alpha3 = config.AGENT.ALPHA_THREE
        self.stride = config.AGENT.STRIDE
        
        # self.my_init_optimizer()
        with torch.no_grad():
            # 4splitDomains
            self.P = torch.eye(2048).type(dtype) # 16MB
            # 10splitTasks
            # self.P_layer4_1_conv1 = torch.eye(2048 * 1 * 1).type(dtype) # 16MB
            # self.P_layer4_1_conv2 = torch.eye(512 * 3 * 3).type(dtype) # 36MB
            # self.P_layer4_1_conv3 = torch.eye(512 * 1 * 1).type(dtype) # 4MB

            self.P_layer4_2_conv1 = torch.eye(2048 * 1 * 1).type(dtype) # 16MB
            self.P_layer4_2_conv2 = torch.eye(512 * 3 * 3).type(dtype) # 36MB
            self.P_layer4_2_conv3 = torch.eye(512 * 1 * 1).type(dtype) # 4MB

        # if self.task_type == '4splitDomains': 
        #     if self.task_count == 1 or self.task_count == 3:
        #         self.config['schedule'][-1] += self.config['schedule'][-1]
        
    # ========================================
    # Rewrite this method, no violation of rules! No change in network structure and forward process. 
    # Modification 1. No use of scheduler.step.
    # Modification 2. we check if now weight is the best ckpt using val_loader after each epoch.
    # ========================================
    def train_model(self, train_loader, val_loader):
        # ========================================
        # Modified here! We use best_acc to store the best acc.
        # ========================================
        best_acc = 0.
        
        for epoch in range(self.config['schedule'][-1]):
            # Config the model and optimizer
            # if self.args.is_main_process:
            self.log.info('Epoch:{0}'.format(epoch))
            if self.args.distributed:
                train_loader.sampler.set_epoch(epoch)
            self.model.train()

            # ========================================
            # Modified here! Commenting code to avoid warnings.
            # ========================================
            # self.scheduler.step(epoch)

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
                # ========================================
                # Modified below! We use validation set to choose the best ckpt.
                # ========================================
                now_acc = self.validation(val_loader)

                if now_acc > best_acc:
                    best_acc = now_acc
                    self.log.info('The best top1 acc of {} is:{}'.format(self.task_count, best_acc))
                    self.log.info('=> Saving model to:' + str(self.save_ckpt_path))
                    torch.save(self.model.state_dict(), self.save_ckpt_path)
                    self.log.info('=> Save Done')

                    self.log.info('=> Saving storage to:' + str(self.save_storage_path))
                    self.save_storage(self.save_storage_path)
                    self.log.info('=> Save storage Done')

    # ========================================
    # Rewrite this method, no violation of rules! No change in network structure and forward process. 
    # Modification 1. Add grad amendments before optimizer.step().
    # ========================================
    def update_model(self, inputs, targets, tasks):
        out = self.forward(inputs)
        loss = self.criterion(out, targets, tasks)
        self.optimizer.zero_grad()
        loss.backward()

        # ========================================
        # Modified below! We use validation set to choose the best ckpt.
        # ========================================
        if self.task_count != 0:
            # Compensate embedding gradients
            for n, w in self.model.named_parameters():
                # if n == 'layer4.1.conv1.weight':
                #     self.pro_weight(self.P_layer4_1_conv1, w, )
                # if n == 'layer4.1.conv2.weight':
                #     self.pro_weight(self.P_layer4_1_conv2, w, )
                # if n == 'layer4.1.conv3.weight':
                #     self.pro_weight(self.P_layer4_1_conv3, w, )

                if n == 'layer4.2.conv1.weight':
                    self.pro_weight(self.P_layer4_2_conv1, w, )
                if n == 'layer4.2.conv2.weight':
                    self.pro_weight(self.P_layer4_2_conv2, w, )
                if n == 'layer4.2.conv3.weight':
                    self.pro_weight(self.P_layer4_2_conv3, w, )

                if n == 'last.All.weight_v' and self.task_type == "4splitDomains": # change
                    self.pro_weight(self.P, w, cnn=False)

        self.optimizer.step()
        return loss.detach(), out

    # ========================================
    # Rewrite this method, no violation of rules! No change in network structure and forward process. 
    # Modification 1. Add seed.
    # Modification 2. load P, which will be used by OWM algorithm.
    # ========================================
    def before_tasks(self, *args, **kwargs):
        if self.reset_optimizer:  # Reset optimizer before learning each task
            # if self.args.is_main_process:
            self.log.info('Optimizer is reset!')
            self.init_optimizer()

        seed = 0
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

        if self.task_count != 0:
            self.load_storage(self.storage_path)
            self.lock_model()

    # ========================================
    # Rewrite this method, no violation of rules! No change in network structure and forward process. 
    # Modification 1. The algorithm needs to forward again to calculate projection matrix P.
    # Modification 2. load best P and ckpt for final file saving in iBatchLearn.
    # ========================================
    def after_tasks(self, train_dataloader):
        self.model.eval()
        with torch.no_grad():
            for i_batch, (input, target, task) in enumerate(train_dataloader):
                if torch.cuda.is_available():
                    input = input.cuda()
                x_list = self.get_x_list(input)
                lamda = i_batch / len(train_dataloader) * 2  # from 0 to 2

                alpha_array = [1.0 * self.alpha0 ** lamda, 1.0 * self.alpha1 ** lamda, 1.0 * self.alpha2 ** lamda, 1.0 * self.alpha3 ** lamda]

                mean_h = torch.mean(torch.matmul(x_list['x_last'].t(), x_list['x_last']))
                std_h = torch.var(x_list['x_last'])
                alpha_array[3] = (mean_h.data.detach() ** 2 + std_h.detach()) ** lamda

                # Compensate embedding gradients
                for n, w in self.model.named_parameters():
                    # if n == 'layer4.1.conv1.weight':
                    #     self.sub_p(self.P_layer4_1_conv1, x_list['x_layer4_1_conv1'], w, alpha=alpha_array[0], stride = self.stride)
                    # if n == 'layer4.1.conv2.weight':
                    #     self.sub_p(self.P_layer4_1_conv2, x_list['x_layer4_1_conv2'], w, alpha=alpha_array[1], stride = self.stride)
                    # if n == 'layer4.1.conv3.weight':
                    #     self.sub_p(self.P_layer4_1_conv3, x_list['x_layer4_1_conv3'], w, alpha=alpha_array[2], stride = self.stride)

                    if n == 'layer4.2.conv1.weight':
                        self.sub_p(self.P_layer4_2_conv1, x_list['x_layer4_2_conv1'], w, alpha = alpha_array[0], stride = self.stride)
                    if n == 'layer4.2.conv2.weight':
                        self.sub_p(self.P_layer4_2_conv2, x_list['x_layer4_2_conv2'], w, alpha = alpha_array[1], stride = self.stride)
                    if n == 'layer4.2.conv3.weight':
                        self.sub_p(self.P_layer4_2_conv3, x_list['x_layer4_2_conv3'], w, alpha = alpha_array[2], stride = self.stride)

                    if n == 'last.All.weight_v' and self.task_type == "4splitDomains": # change
                        self.sub_p(self.P, x_list['x_last'], w, alpha=alpha_array[3], cnn = False)  

        model_state = torch.load(self.save_ckpt_path)
        self.model.load_state_dict(model_state)
        self.log.info('=> Load best model Done')

    # ========================================
    # Rewrite this method, no violation of rules! No change in network structure and forward process. 
    # ========================================
    def before_epoch(self, *args, **kwargs):
        if self.task_count != 0: 
            self.model.eval()

    # ========================================
    # Rewrite this method, no violation of rules! No change in network structure and forward process. 
    # ========================================
    def collect_memoey(self) -> dict:
        '''
        Return a variable of dictionary type,
        each value in the dictionary should be of type ``float32".
        '''
        memory = {
            'state_dict': self.model.state_dict(),
            'p': self.P ,
            # 'P_layer4_1_conv1': self.P_layer4_1_conv1,
            # 'P_layer4_1_conv2': self.P_layer4_1_conv2,
            # 'P_layer4_1_conv3': self.P_layer4_1_conv3,
            'P_layer4_2_conv1': self.P_layer4_2_conv1,
            'P_layer4_2_conv2': self.P_layer4_2_conv2,
            'P_layer4_2_conv3': self.P_layer4_2_conv3,
            'optimizer' : self.optimizer.state_dict(),
            }
        return memory

    # ========================================
    # Rewrite this method, no violation of rules! No change in network structure and forward process. 
    # ========================================
    def allocate_memory(self, storage):
        print('=> Loading optimizer')
        self.optimizer.load_state_dict(storage['optimizer'])
        print('=> Load optimizer Done')

        print('=> Loading P')
        self.P = storage['p']
        print('=> Load P Done')

        print('=> Loading P_layer4_1, P_layer4_2')
        # self.P_layer4_1_conv1 = storage['P_layer4_1_conv1']
        # self.P_layer4_1_conv2 = storage['P_layer4_1_conv2']
        # self.P_layer4_1_conv3 = storage['P_layer4_1_conv3']
        self.P_layer4_2_conv1 = storage['P_layer4_2_conv1']
        self.P_layer4_2_conv2 = storage['P_layer4_2_conv2']
        self.P_layer4_2_conv3 = storage['P_layer4_2_conv3']
        print('=> Load P_layer4_1, P_layer4_2 Done')

    # Add this method, doesn't exist in CLTrainer
    def sub_p(self, p, x, w, alpha=1.0, cnn=True, stride=1):
        with torch.no_grad():
            if cnn:
                pass
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
                # w.grad.data = torch.mm(w.grad.data.view(F, -1), torch.t(p.data)).view_as(w)
            else:
                r = x
                k = torch.mm(p, torch.t(r))
                p.sub_(torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
                # w.grad.data = torch.mm(w.grad.data, torch.t(p.data))

    # Add this method, doesn't exist in CLTrainer
    def get_x_list(self, x):
        x_list = {}
        if self.task_type == "10splitTasks" or self.task_type == "4splitDomains":
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)

            x = self.model.layer4[0](x)
            x = self.model.layer4[1](x)

            # identity1 = x
            # x_list['x_layer4_1_conv1'] = torch.mean(x, 0, True)
            # x = self.model.layer4[1].conv1(x)
            # x = self.model.layer4[1].bn1(x)
            # x = self.model.layer4[1].relu(x)

            # x_list['x_layer4_1_conv2'] = torch.mean(x, 0, True)
            # x = self.model.layer4[1].conv2(x)
            # x = self.model.layer4[1].bn2(x)
            # x = self.model.layer4[1].relu(x)

            # x_list['x_layer4_1_conv3'] =  torch.mean(x, 0, True)
            # x = self.model.layer4[1].conv3(x)
            # x = self.model.layer4[1].bn3(x)

            # out = x + identity1
            # x = self.model.layer4[1].relu(out)

            identity2 = x
            x_list['x_layer4_2_conv1'] = torch.mean(x, 0, True)
            out = self.model.layer4[2].conv1(x)
            out = self.model.layer4[2].bn1(out)
            out = self.model.layer4[2].relu(out)

            x_list['x_layer4_2_conv2'] = torch.mean(out, 0, True)
            out = self.model.layer4[2].conv2(out)
            out = self.model.layer4[2].bn2(out)
            out = self.model.layer4[2].relu(out)

            x_list['x_layer4_2_conv3'] = torch.mean(out, 0, True)
            out = self.model.layer4[2].conv3(out)
            out = self.model.layer4[2].bn3(out)
            out += identity2

            out = self.model.layer4[2].relu(out)

            out = self.model.avgpool(out)
            out = torch.flatten(out, 1)
            x_list['x_last'] = torch.mean(out, 0, True) # change

        else:
            print("get invalid task_type!")

        return x_list

    # Add this method, doesn't exist in CLTrainer
    def lock_model(self,):
        ct = 0
        for child in self.model.children():
            ct = ct + 1
            if ct == 10:
                print("only train model.last ")
                for param in child.parameters():
                    param.requires_grad = True
            elif ct == 8:
                for n, param in child.named_parameters():
                    # if n == '2.conv1.weight' or n == '2.conv2.weight' or n == '2.conv3.weight' or n == '1.conv1.weight' or n == '1.conv2.weight' or n == '1.conv3.weight':
                    if n == '2.conv1.weight' or n == '2.conv2.weight' or n == '2.conv3.weight':
                        param.requires_grad = True # Change
                    else:
                        param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = False

    # Add this method, doesn't exist in CLTrainer
    def pro_weight(self, p, w, cnn=True, ):
        pass
        if cnn:
            F, _, HH, WW = w.shape
            w.grad.data = torch.mm(w.grad.data.view(F, -1), torch.t(p.data)).view_as(w)
        else:
            w.grad.data = torch.mm(w.grad.data, torch.t(p.data))