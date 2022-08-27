import numpy as np
import math
import parameters_ym as gol
import uproot as up
import numba as nb
from numba import float64


class electronResponse(object):

    def __init__(self) -> None:
        """
        load quenching nonlinearity curve clusters into _global_kB_dict
        """
        gol.set_run_mode("cpu")

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
    @nb.guvectorize(["float64[:], int64[:], float64[:], float64, float64[:]"],
                    "(n), (n), (m), ()->(n)",
                    target="parallel",
                    nopython=True)
    def get_Nsct(E, idE, nonl, Ysct, Nsct):
        for i in range(E.shape[0]):
            nl = nonl[idE[i]]
            Nsct[i] = nl * Ysct * E[i]

    @staticmethod
    #@profile
    @nb.vectorize([float64(float64, float64, float64, float64, float64)],
                  target="parallel",
                  nopython=True)
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
    # @np.vectorize
    @nb.vectorize([float64(float64, float64, float64, float64)],
                  target="parallel",
                  nopython=True)
    def get_Nsigma(N, a, b, n):
        """
        calculate NPE sigma value
        input: NPE
        output: sigma_NPE
        """
        return math.sqrt(a**2 * N + b**2 * N**n)
