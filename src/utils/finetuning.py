import torch  
import os
import gc
import torchaudio 
import pandas 

torch.set_num_threads(16)

def clear_gpu_cache():
    # clear the GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



