# decision on using GPU or CPU
import os
use_gpu = True # set True if GPU is available
if not use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np

nu      = 0.01 / np.pi
nu_range = [1.0 / np.pi, 0.1 / np.pi, 0.01 / np.pi, 0.001 / np.pi, 0.0001 / np.pi]
N_f     = 8000
N_bc    = 100
N_ic    = 200
Epochs = 15000
LAYERS      = [2, 20, 20, 20, 20, 1]
LAYERS_PINO = [3] + LAYERS[1:]   # (x, t, log10(nu)) → same hidden layers → u

# PI-FNO grid and architecture
NX_FNO    = 128   # spatial grid points (must be even)
NT_FNO    = 50    # time grid points
FNO_WIDTH = 32    # channel width inside FNO layers
FNO_MODES = 16    # number of Fourier modes to keep
FNO_DEPTH = 4     # number of FNO blocks