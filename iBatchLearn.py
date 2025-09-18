import argparse
import os
import os.path as osp
from utils.config import get_config
from utils.logger import create_logger, get_logger
from utils.utils import set_seed, set_ddp
import json
import agents
import datasets.base
import torch
from utils.dist_utils import SequentialDistributedSampler, get_world_size
import pickle
from utils.metric import compute_acc_bwt
from collections import OrderedDict
import numpy as np
import torch.distributed as dist
import yaml


def _update_config_from_file(config, cfg_file, suffix):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(config, os.path.join(os.path.dirname(cfg_file), cfg))
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.LOGGER_PATH = os.path.join('outputs', config.LOGGER_PATH,
                                      config.AGENT.TYPE + '-' + config.AGENT.NAME
                                      + f'-{suffix}')

    config.freeze()


def compute_mem_score(mems, init_model_path):
    storage_mem = np.array(mems).max()
    bench_mem = osp.getsize(init_model_path) / float(1024 * 1024)
    mem_ratio = float(storage_mem) / bench_mem
    mem_score = 0.5 * np.maximum(mem_ratio - 5.0, 0)
    return mem_score

def make_agent(config, args, ckpt_path):
    logger = get_logger(name=config.DATASET.NAME)
    if config.DOMAIN_INCR:
        # domain incremental
        out_dim = {'All': config.DATASET.NUM_CLASSES}
    else:
        # task incremental
        split_size = config.DATASET.NUM_CLASSES // config.DATASET.NUM_TASKS
        out_dim = {}
        for i in range(config.DATASET.NUM_TASKS):
            out_dim.update({str(i): split_size})

    agent = agents.__dict__[config.AGENT.TYPE].__dict__[config.AGENT.NAME](config, args, logger, out_dim, ckpt_path)

    logger.info(agent.model)
    logger.info(f'#parameter of model:{agent.count_parameter()}')
    return agent


def pred(config, args):
    logger = get_logger(name=config.DATASET.NAME)
    i = args.task_count
    agent = make_agent(config, args, args.ckpt_path)

    if not args.test:
        dir_ = os.path.join(config.LOGGER_PATH, 'checkpoints')
        if not os.path.exists(dir_):
            os.mkdir(dir_)

        acc_table_path = os.path.join(dir_, 'acc_table_val.pth')
        mem_path = os.path.join(dir_, 'mem_path.pth')

        if i > 0:
            mem = agent.load_storage(args.storage_path)

            with open(acc_table_path, 'rb') as f:
                acc_table_val = pickle.load(f)
            with open(mem_path, 'rb') as f:
                mems = pickle.load(f)
            mems.append(mem)
            mem_score = compute_mem_score(mems, args.init_path)
            logger.info(f'memory score: {mem_score}')
        else:
            acc_table_val = OrderedDict()
            mems = []

        train_name = str(i)

        if args.is_main_process:
            logger.info(f'====================== {train_name} =======================')
        train_dataset = datasets.base.__dict__[
            "get_" + config.DATASET.NAME](config.DATASET.ROOT, 'train', train_name)
        val_dataset = datasets.base.__dict__[
            "get_" + config.DATASET.NAME](config.DATASET.ROOT, 'val', train_name)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            batch_size = int(config.DATASET.BATCHSIZE / get_world_size())
            val_sampler = SequentialDistributedSampler(val_dataset, batch_size=batch_size)
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True
            batch_size = config.DATASET.BATCHSIZE
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=shuffle,
                                                    sampler = train_sampler,
                                                    num_workers=config.DATASET.NUM_WORKERS,
                                                    pin_memory = True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    sampler = val_sampler,
                                                    num_workers=config.DATASET.NUM_WORKERS,
                                                    pin_memory = True)
        # Train
        agent.learn_tasks(train_loader, val_loader)
        agent.save_model(args.save_ckpt_path)

        # For validation
        if args.distributed:
            dist.barrier()
        acc_table_val[train_name] = OrderedDict()
        for j in range(i+1):
            val_name = str(j)
            if args.is_main_process:
                logger.info(f'validation split name:{val_name}')
            val_data = datasets.base.__dict__[
                "get_" + config.DATASET.NAME](config.DATASET.ROOT, 'val', val_name)
            if args.distributed:
                val_sampler = SequentialDistributedSampler(val_data, batch_size=batch_size)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(val_data,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        sampler = val_sampler,
                                                        num_workers=config.DATASET.NUM_WORKERS,
                                                        pin_memory = True)
            acc_table_val[val_name][train_name] = agent.validation(val_loader)
            if args.distributed:
                dist.barrier()

        agent.save_storage(args.save_storage_path)
        with open(acc_table_path, 'wb') as f:
            pickle.dump(acc_table_val, f)
        with open(mem_path, 'wb') as f:
            pickle.dump(mems, f)
        acc, bwt= compute_acc_bwt(acc_table_val, num_tasks=i+1)
        if args.is_main_process:
            logger.info(f"Acc:{acc}; BWT:{bwt};")

    # For test
    else:
        if args.distributed:
            dist.barrier()
            batch_size = int(config.DATASET.BATCHSIZE / get_world_size())
        else:
            batch_size = config.DATASET.BATCHSIZE
        predicts = []
        for j in range(i + 1):
            test_name = str(j)
            logger.info(f'test split name:{test_name}')
            test_data = datasets.base.__dict__[
                "get_" + config.DATASET.NAME](config.DATASET.ROOT, 'test', test_name)
            if args.distributed:
                test_sampler = SequentialDistributedSampler(test_data, batch_size=batch_size)
            else:
                test_sampler = None
            test_loader = torch.utils.data.DataLoader(test_data,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      sampler=test_sampler,
                                                      pin_memory=True,
                                                      num_workers=config.DATASET.NUM_WORKERS)
            predict = agent.test(test_loader).cpu().tolist()
            predicts.append(predict)
            if args.distributed:
                dist.barrier()

        with open(args.dest_path, 'wb') as f:
            pickle.dump(predicts, f)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--user_cfg', type=str, required=True)
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--task_count', default=0, type=int)
    parser.add_argument('--init_path', default=None, type=str)
    parser.add_argument('--ckpt_path', default=None, type=str)
    parser.add_argument('--save_ckpt_path', default=None, type=str)
    parser.add_argument('--storage_path', default=None, type=str)
    parser.add_argument('--save_storage_path', default=None, type=str)
    parser.add_argument('--dest_path', default=None, type=str)
    parser.add_argument('--suffix', default=None, type=str)
    args = parser.parse_args()

    config = get_config()
    _update_config_from_file(config, args.user_cfg, args.suffix)
    _update_config_from_file(config, args.cfg, args.suffix)

    if 'WORLD_SIZE' in os.environ:
        args.distributed = True
        seed = config.SEED + args.local_rank
        os.environ["NCCL_SOCKET_IFNAME"] = args.socket_ifname
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        args.is_main_process = set_ddp(args)
    else:
        args.distributed = False
        seed = config.SEED
        args.is_main_process = True

    set_seed(seed)

    if not os.path.exists(config.LOGGER_PATH) and args.is_main_process:
        os.makedirs(config.LOGGER_PATH)
    logger = create_logger(config.LOGGER_PATH, filename=config.DATASET.NAME + '.txt', name=config.DATASET.NAME)

    if args.is_main_process:
        pa = osp.join(config.LOGGER_PATH, 'config.json')
        with open(pa, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {pa}")

    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    pred(config, args)

if __name__ == '__main__':
    main()