# decision on using GPU or CPU
import os
use_gpu = True # set True if GPU is available
if not use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np

nu      = 0.01 / np.pi
N_f     = 10000
N_bc    = 100
N_ic    = 200
Epochs = 20000
LAYERS  = [2, 40, 40, 40, 40, 1]