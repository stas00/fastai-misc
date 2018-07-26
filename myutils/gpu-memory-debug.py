# get CPU/GPU RAM usage reports like:
# Gen RAM Free: 11.6 GB  | Proc size: 666.0 MB
# GPU RAM Free: 566MB | Used: 10873MB | Util  95% | Total 11439MB

# needed on google colab
!ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi

# memory footprint support libraries/code
!pip install gputil
!pip install psutil
!pip install humanize
import psutil
import humanize
import os
import GPUtil as GPU
GPUs = GPU.getGPUs()
# XXX: only one GPU on Colab and isn’t guaranteed
gpu = GPUs[0]
def printm():
 process = psutil.Process(os.getpid())
 print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " I Proc size: " + humanize.naturalsize( process.memory_info().rss))
 print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))

printm()


