import os 
import random
import numpy as np 
import torch 
from utils.dist_utils import synchronize, is_main_process

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_ddp(args):
    if args.distributed:
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method=args.init_method,
            rank=args.local_rank,
            world_size=world_size
        )
        synchronize()
    
        
        return is_main_process()
