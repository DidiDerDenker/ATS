# Imports
from __future__ import print_function

import math
import pickle
import torch.distributed

from others.logging import logger


# Methods
def is_master(gpu_ranks, device_id):
    return gpu_ranks[device_id] == 0


def multi_init(device_id, world_size, gpu_ranks):
    print(gpu_ranks)
    dist_init_method = "tcp://localhost:10000"
    dist_world_size = world_size

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=dist_init_method,
        world_size=dist_world_size,
        rank=gpu_ranks[device_id]
    )

    gpu_rank = torch.distributed.get_rank()

    if not is_master(gpu_ranks, device_id):
        logger.disabled = True

    return gpu_rank


def all_reduce_and_rescale_tensors(tensors, rescale_denom, buffer_size=10485760):
    """
    All-reduce and rescale tensors in chunks of the specified size.

    Args:
        tensors: list of Tensors to all-reduce
        rescale_denom: denominator for rescaling summed Tensors
        buffer_size: all-reduce chunk size in bytes
    """

    buffer_t = tensors[0].new(math.ceil(buffer_size / tensors[0].element_size())).zero_()
    buffer = []

    def all_reduce_buffer():
        # Copy tensors into buffer_t
        offset = 0

        for t in buffer:
            numel = t.numel()
            buffer_t[offset:offset+numel].copy_(t.view(-1))
            offset += numel

        # All-reduce and rescale
        torch.distributed.all_reduce(buffer_t[:offset])
        buffer_t.div_(rescale_denom)

        # Copy all-reduced buffer back into tensors
        offset = 0

        for t in buffer:
            numel = t.numel()
            t.view(-1).copy_(buffer_t[offset:offset+numel])
            offset += numel

    filled = 0

    for t in tensors:
        sz = t.numel() * t.element_size()

        if sz > buffer_size:
            # Tensor is bigger than buffer, all-reduce and rescale directly
            torch.distributed.all_reduce(t)
            t.div_(rescale_denom)

        elif filled + sz > buffer_size:
            # Buffer is full, all-reduce and replace buffer with grad
            all_reduce_buffer()
            buffer = [t]
            filled = sz

        else:
            # Add tensor to buffer
            buffer.append(t)
            filled += sz

    if len(buffer) > 0:
        all_reduce_buffer()


def all_gather_list(data, max_size=4096):
    """ Gathers arbitrary data from all nodes into a list. """
    world_size = 4 # TODO: Use GPU, e.g. world_size = torch.distributed.get_world_size()

    if not hasattr(all_gather_list, "_in_buffer") or max_size != all_gather_list._in_buffer.size():
        all_gather_list._in_buffer = torch.cuda.ByteTensor(max_size)
        all_gather_list._out_buffers = [torch.cuda.ByteTensor(max_size) for i in range(world_size)]

    in_buffer = all_gather_list._in_buffer
    out_buffers = all_gather_list._out_buffers

    enc = pickle.dumps(data)
    enc_size = len(enc)

    if enc_size + 2 > max_size:
        raise ValueError("Encoded data exceeds max_size: {}".format(enc_size + 2))

    assert max_size < 255 * 256
    in_buffer[0] = enc_size // 255
    in_buffer[1] = enc_size % 255
    in_buffer[2:enc_size+2] = torch.ByteTensor(list(enc))

    torch.distributed.all_gather(out_buffers, in_buffer.cuda())
    results = []

    for i in range(world_size):
        out_buffer = out_buffers[i]
        size = (255 * out_buffer[0].item()) + out_buffer[1].item()
        bytes_list = bytes(out_buffer[2:size+2].tolist())
        result = pickle.loads(bytes_list)
        results.append(result)

    return results
