#import numba as nb
from numba import float64
from timebudget import timebudget
import numpy as np

a = np.ones((1000, 1000), np.int64) * 5 
b = np.ones((1000, 1000), np.int64) * 10 
c = np.ones((1000, 1000), np.int64) * 15

@timebudget
def add_arrays(a, b, c): 
    return np.sqrt(a, b, c) 

@timebudget
@nb.njit 
def add_arrays_numba(a, b, c): 
    return np.sqrt(a, b, c) 


if __name__ == "__main__":

    add_arrays_numba(a, b, c)

    add_arrays_numba(a, b, c)
    add_arrays(a, b, c)

    from electronResponse_ym import electronResponse
    electronResponse.get_Ncer()



    import parameters_ym as p
    p.get_kB_value()
