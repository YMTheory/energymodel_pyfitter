import numpy as np
import math
import parameters_ym as gol
import uproot as up
import numba as nb
from numba import int32, int64, float32, float64
from numba import cuda


class electronResponse(object):

    def __init__(self) -> None:
        """
        load quenching nonlinearity curve clusters into _global_kB_dict
        """
        gol.set_run_mode("cuda")

        print("------ Initialzed electron response -----")
        #quenchNL_file = "/Volumes/home/Data/EnergyModel/Quench_NumInt.root"
        quenchNL_file = "/hpcfs/juno/junogpu/miaoyu/energy_model/data/Quench_NumInt.root"
        try:
            fquenchNL = up.open(quenchNL_file)

            for hname in fquenchNL._keys_lookup:
                hquenchNL = fquenchNL[f"{hname}"]
                gol.set_kB_value(hname, hquenchNL.to_numpy()[0])

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Quenching nonlinearity file {quenchNL_file} dose not exist! The Fitter can not be executed furthermore :("
            )

        # a nominal initial values
        electronResponse.quenchNL = gol.get_kB_value("kB65")

    @staticmethod
    def update():
        kB = gol.get_fitpar_value("kB")
        kBidx = int(kB * 10000)
        electronResponse.quenchNL = gol.get_kB_value(f"kB{kBidx}")

    @staticmethod
    def get_nonl():
        return electronResponse.quenchNL

    @staticmethod
    @cuda.jit()
    def get_Nsct_cuda(E, idE, nonl, Ysct, Nsct):
        """
        CUDA jit acceleration test
        """
        x, y = cuda.grid(2)
        if x < Nsct.shape[0] and y < Nsct.shape[1]:
            nl = nonl[idE[x, y]]
            Nsct[x, y] = nl * Ysct * E[x, y]

    ## @staticmethod
    ## @cuda.jit
    ## def get_Nsct_cuda_sharedmem(E, idE, nonl, Nsct):
    ##     """
    ##     kernel function: CUDA jit compilation, using shared memory.
    ##     """            
    ##     # thread number in each block
    ##     BLOCK_SIZE = 16
    ##     ## allocate shared memory for the current block
    ##     sE = cuda.shared.array(shape=(BLOCK_SIZE, BLOCK_SIZE), dtype=float64)
    ##     sidE = cuda.shared.array(shape=(BLOCK_SIZE, BLOCK_SIZE), dtype=int32)
    ##     ## relative position for the current thread in the block
    ##     tx = cuda.threadIdx.x
    ##     ty = cuda.threadIdx.y
    ##     ## absolute position for the current thread
    ##     row = tx + cuda.blockDim.x * cuda.blockIdx.x
    ##     col = ty + cuda.blockDim.y * cuda.blockIdx.y
    ##     
    ##     if row > Nsct.shape[0] and col > Nsct.shape[1]:
    ##         return
    ##     
    ##     for m in range(math.ceil(E.shape[0] / BLOCK_SIZE)):
    ##         for n in range(math.ceil(E.shape[1] / BLOCK_SIZE)):
    ##             sE[tx, ty] = E[tx + m * BLOCK_SIZE, ty + n * BLOCK_SIZE]
    ##             sidE[tx, ty] = idE[tx + m * BLOCK_SIZE, ty + n * BLOCK_SIZE]
    ##             cuda.syncthreads()
            
            

    @staticmethod
    @nb.guvectorize(
        ["void(float64[:], int64[:], float64[:], float64, float64[:])"],
        "(n), (n), (l), ()->(n)",
        target="cuda")
    def get_Nsct(E, idE, nonl, Ysct, Nsct):
        """
        calculate scintillation photon number with numba
        input: true deposited energy
        output: scintillation photon number
        """
        for i in range(E.shape[0]):
            nl = nonl[idE[i]]
            Nsct[i] = nl * Ysct * E[i]

    @staticmethod
    @nb.vectorize([float64(float64, float64, float64, float64, float64)],
                  target="cuda")
    def get_Ncer(E, p0, p1, p2, E0):
        """
        calculate Cherenkov photon number
        input: true deposited energy
        output: Cherenkov photon number
        """
        E = E - E0
        if E < 0:
            return 0
        else:
            return p0 * E**2 / (E + p1 * math.exp(-p2 * E))

    @staticmethod
    @nb.vectorize([float64(float64, float64, float64, float64)], target="cuda")
    def get_Nsigma(N, a, b, n):
        """
        calculate NPE sigma value
        input: NPE
        output: sigma_NPE
        """
        return math.sqrt(a**2 * N + b**2 * N**n)
