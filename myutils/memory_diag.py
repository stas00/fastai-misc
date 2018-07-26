# get CPU/GPU RAM usage reports like:
# Gen RAM Free: 11.6 GB  | Proc size: 666.0 MB
# GPU RAM Free: 566MB | Used: 10873MB | Util  95% | Total 11439MB

# import sys
# sys.path.append('/home/stas/fast.ai')
# from myutils.memory_diag import printm

import psutil
import humanize
hs = humanize.naturalsize
import os
import GPUtil as GPU
GPUs = GPU.getGPUs()
process = psutil.Process(os.getpid())

# XXX: there could be more than one GPU (or none!)
gpu = GPUs[0]

def printm():
    """Print memory usage (not exact due to pytorch memory caching)"""
    print("Gen RAM Free {0:>7s} | Proc size {1}".format(
        hs(psutil.virtual_memory().available), 
        hs(process.memory_info().rss))) 
    print("GPU RAM Free {0:>7s} | Used {1} | Util {2:2.1f}% | Total {3}".format(
        hs(gpu.memoryFree*1024**2), hs(gpu.memoryUsed*1024**2), 
        gpu.memoryUtil*100, hs(gpu.memoryTotal*1024**2)))

#printm()


def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert(isinstance(size, torch.Size))
    return " × ".join(map(str, size))

# pytorch tensors dump
def dump_tensors(gpu_only=True):
    """Prints a list of the Tensors being tracked by the garbage collector."""
    import gc
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print("%s:%s%s %s" % (type(obj).__name__, 
                                          " GPU" if obj.is_cuda else "",
                                          " pinned" if obj.is_pinned else "",
                                          pretty_size(obj.size())))
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    print("%s → %s:%s%s%s%s %s" % (type(obj).__name__, 
                                                   type(obj.data).__name__, 
                                                   " GPU" if obj.is_cuda else "",
                                                   " pinned" if obj.data.is_pinned else "",
                                                   " grad" if obj.requires_grad else "", 
                                                   " volatile" if obj.volatile else "",
                                                   pretty_size(obj.data.size())))
                    total_size += obj.data.numel()
        except Exception as e:
            pass        
    print("Total size:", total_size)

