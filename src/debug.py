import os
import torch.distributed as dist

# Set environment variables explicitly
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '28000'

# Initialize the process group using use_libuv=0
dist.init_process_group(
    backend='nccl',  # Or 'gloo' if using CPU
    init_method='tcp://127.0.0.1:28000?use_libuv=0',  # Keep use_libuv=0 parameter
    world_size=1,  # Total number of processes
    rank=0         # Rank of the current process
)

print("NCCL backend initialized successfully with use_libuv=0.")

# Cleanup
dist.destroy_process_group()
