import os,sys
import numpy as np
import torch
import utils
from torchvision import datasets,transforms
import shift

def get(seed=0, pc_valid=0.10, t_num = 2):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    # CIFAR10
    
    if (not os.path.isfile('../../data/binary_cifar/t_num_' + str(t_num) + 'data' + str(10//t_num) + 'trainx.bin')):    
        if (not os.path.isdir('../../data/binary_cifar/')):
            os.makedirs('../../data/binary_cifar')

        mean = [x / 255 for x in [125.3, 123.0, 113.9]] # what? normalize the img
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat={}
        # dat['train']=datasets.CIFAR10('../../data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['train']=shift.CIFAR10('../../data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        print('shift Done !')
        dat['test']=datasets.CIFAR10('../../data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        # dat['test']=shift.CIFAR10('../../data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        # print(dat['train'].data[0])
        # print(type(dat['train'].data[0]))
        # print(type(dat['train'].targets[0]))
        # print(dat['train'].data[0].max())
        # raise ValueError("Dev mode")
        for t in range(10//t_num):
            data[t] = {} # dict for each task, add different property
            data[t]['name'] = 'cifar10-' + str(t_num*t) + '-' + str(t_num*(t+1)-1) # dict key is 'name', value is string
            data[t]['ncla'] = t_num # dict key is 'number of class', value is int number
            '''
            The following paragraph is spliting dataset for each task.
            '''
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_num*t, t_num*(t+1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 10 // t_num
        data[t] = {} # Recording all task
        data[t]['name'] = 'cifar10-all'
        data[t]['ncla'] = 10
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []} # data[task][train/test] is a dict, optional x & y
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys(): # all the keys in dict data, i.e. task number 
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2]) # stack all sample into one tensor
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)  # save all targets into one tensor
                torch.save(data[t][s]['x'], '../../data/binary_cifar/t_num_' + str(t_num) + 'data' + str(t) + s + 'x.bin')
                torch.save(data[t][s]['y'], '../../data/binary_cifar/t_num_' + str(t_num) + 'data' + str(t) + s + 'y.bin')
    
    # Load binary files
    data = {}
    ids = list(np.arange((10 // t_num) + 1))
    print('Task order =', ids)
    for i in range((10 // t_num) + 1):
        data[i] = dict.fromkeys(['name','ncla','train','test']) 
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load('../../data/binary_cifar/t_num_'+str(t_num)+'data'+str(ids[i])+s+'x.bin')
            data[i][s]['y']=torch.load('../../data/binary_cifar/t_num_'+str(t_num)+'data'+str(ids[i])+s+'y.bin')
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy())) # count the actually class
        data[i]['name'] = 'cifar10->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla'] = n

    return data, taskcla[:10//data[0]['ncla']], size

def get_cifar10(seed=0, pc_valid=0.10, t_num = 2, if_atk = False, perc = 0.0):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    # CIFAR10

    mean = [x / 255 for x in [125.3, 123.0, 113.9]] # what? normalize the img
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    dat={}
    if if_atk == True:
        dat['train']=shift.CIFAR10('../../data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]), perc=perc)
        print('Shift Done !')
    else:
        dat['train']=datasets.CIFAR10('../../data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test']=datasets.CIFAR10('../../data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    
    # raise ValueError("Dev mode")
    for t in range(10//t_num):
        data[t] = {} # dict for each task, add different property
        data[t]['name'] = 'cifar10-' + str(t_num*t) + '-' + str(t_num*(t+1)-1) # dict key is 'name', value is string
        data[t]['ncla'] = t_num # dict key is 'number of class', value is int number
        '''
        The following paragraph is spliting dataset for each task.
        '''
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                if label in range(t_num*t, t_num*(t+1)):
                    data[t][s]['x'].append(image)
                    data[t][s]['y'].append(label)
    
    t = 10 // t_num
    data[t] = {} # Recording all task
    data[t]['name'] = 'cifar10-all'
    data[t]['ncla'] = 10
    for s in ['train', 'test']:
        loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
        data[t][s] = {'x': [], 'y': []} # data[task][train/test] is a dict, optional x & y
        for image, target in loader:
            label = target.numpy()[0]
            data[t][s]['x'].append(image)
            data[t][s]['y'].append(label)

    # "Unify" and save
    for t in data.keys(): # all the keys in dict data, i.e. task number 
        for s in ['train', 'test']:
            data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2]) # stack all sample into one tensor
            data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)  # save all targets into one tensor

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla'] = n

    return data, taskcla[:10//data[0]['ncla']], size

def get_cifar100(seed=0, pc_valid=0.10, t_num = 2, if_atk = False, perc = 0.0):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    # CIFAR100

    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2,  65.4, 70.4]]
    
    dat={}
    if if_atk == True:
        dat['train']=shift.CIFAR100('../../data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]), perc=perc)
        print('Shift Done !')
    else:
        dat['train']=datasets.CIFAR100('../../data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test']=datasets.CIFAR100('../../data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    
    # raise ValueError("Dev mode")
    for t in range(100//t_num):
        data[t] = {} # dict for each task, add different property
        data[t]['name'] = 'cifar100-' + str(t_num*t) + '-' + str(t_num*(t+1)-1) # dict key is 'name', value is string
        data[t]['ncla'] = t_num # dict key is 'number of class', value is int number
        '''
        The following paragraph is spliting dataset for each task.
        '''
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                if label in range(t_num*t, t_num*(t+1)):
                    data[t][s]['x'].append(image)
                    data[t][s]['y'].append(label)
    
    t = 100 // t_num
    data[t] = {} # Recording all task
    data[t]['name'] = 'cifar100-all'
    data[t]['ncla'] = 100
    for s in ['train', 'test']:
        loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
        data[t][s] = {'x': [], 'y': []} # data[task][train/test] is a dict, optional x & y
        for image, target in loader:
            label = target.numpy()[0]
            data[t][s]['x'].append(image)
            data[t][s]['y'].append(label)

    # "Unify" and save
    for t in data.keys(): # all the keys in dict data, i.e. task number 
        for s in ['train', 'test']:
            data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2]) # stack all sample into one tensor
            data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)  # save all targets into one tensor

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla'] = n

    return data, taskcla[:100//data[0]['ncla']], size