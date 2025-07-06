import torch
import torch.distributed as dist
import os

def test_all_reduce():
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )
    tensor = torch.ones(1).cuda()
    dist.all_reduce(tensor)
    print(f"Rank {dist.get_rank()}: AllReduce结果 {tensor}")

if __name__ == "__main__":
    test_all_reduce()