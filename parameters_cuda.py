import parameters_ym as gol
import numpy as np
from numba import cuda

global d_quenchNL


def copy_quenchNL_to_device():
    """
    If executing on CUDA, copy the electron quenchNL firstly to the device.
    """
    qnl = np.zeros(gol.get_kB_value("kB45").shape)  # hard-coding here
    for key, val in gol._global_kB_dict.items():
        tmp = np.vstack((qnl, val))
        qnl = tmp
    qnl = np.delete(qnl, 0, axis=0)
    
    global d_quenchNL
    d_quenchNL = cuda.to_device(qnl)

    
def get_device_quenchNL(kB):
    KBMIN = 45
    kbidx = int(kB*10000) - KBMIN
    return d_quenchNL[kbidx]