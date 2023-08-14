import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import tqdm
from utils.parser_argument import argparse_config

argparse_config()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size = world_size)

def cleanup():
    dist.destroy_process_group()

class SampleDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 10
    
    def __getitem__(self, idx):
        x = torch.rand(20, 10)
        y = torch.rand(20, 5)
        return x, y
    

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))
    

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # Dataset
    dataset = SampleDataset()

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank = rank,
        shuffle = True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler
    )

    # create model and move it to GPu with id rank
    model = ToyModel().to(rank)

    if args.fp16:
        model = model.half()

    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr = 0.001)

    for epoch in tqdm.tqdm(range(args.epochs)):
        for i, (x, y) in enumerate(dataloader):
            if args.fp16:
                x = x.half()

            optimizer.zero_grad()
            ouptuts = ddp_model(x.to(rank))
            labels = y.to(rank)
            loss_fn(ouptuts, labels).backward()
            optimizer.step()

            if i == 5:
                break
  
    cleanup()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size, ),
             nprocs = world_size,
             join=True)
    
if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()

    world_size = n_gpus

    run_demo(demo_basic, world_size)