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
        print("------ Initialzed electron response -----")
        #quenchNL_file = "/Volumes/home/Data/EnergyModel/Quench_NumInt.root"
        quenchNL_file = "/hpcfs/juno/junogpu/miaoyu/energy_model/data/Quench_NumInt.root"
        try:
            fquenchNL = up.open(quenchNL_file)

            Emin, Emax, Estep = 5e-4, 14.9995, 1e-3
            gol.set_quenchE(np.arange(Emin, Emax, Estep))

            for hname in fquenchNL._keys_lookup:
                hquenchNL = fquenchNL[f"{hname}"]
                gol.set_kB_value(hname, hquenchNL.to_numpy()[0])    

        except FileNotFoundError:
            raise FileNotFoundError(f"Quenching nonlinearity file {quenchNL_file} dose not exist! The Fitter can not be executed furthermore :(")

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
    def _get_Nsct_cuda(E, idE, nonl, Ysct, Nsct):
        """
        CUDA jit acceleration test
        """
        ## for i in range(E.shape[0]):
        ##     for j in range(E.shape[1]):
        ##         nl = nonl[idE[i, j]]
        ##         Nsct[i, j] = nl * Ysct * E[i, j]

        x, y = cuda.grid(2)
        if x < Nsct.shape[0] and y < Nsct.shape[1]:
            nl = nonl[idE[x, y]]
            Nsct[x, y] = nl * Ysct * E[x, y]




    @staticmethod
    @nb.guvectorize(["float64[:], int64[:], float64[:], float64, float64[:]"], "(n), (n), (m), ()->(n)", target="parallel", nopython=True)
    #@nb.guvectorize(["void(float64[:, :], int64[:, :], float64[:], float64, float64[:, :])"], "(m, n), (m, n), (l), ()->(m, n)", target="cuda")
    def _get_Nsct(E, idE, nonl, Ysct, Nsct):
        """
        calculate scintillation photon number with numba
        input: true deposited energy
        output: scintillation photon number
        """
        ### codes for non-cuda numba version
        for i in range(E.shape[0]):
            nl = nonl[idE[i]]
            Nsct[i] = nl * Ysct * E[i]
        ###

        #### codes for cuda version
        #for i in range(E.shape[0]):
        #    for j in range(E.shape[1]):
        #        nl = nonl[idE[i, j]]
        #        Nsct[i, j] = nl * Ysct * E[i, j]


    @np.vectorize
    #@profile
    def get_Nsct(E, kB, Ysct):
        idx, idy = int(kB*10000), int(E * 1000)
        nonl = gol.get_kB_value(f"kB{idx}")[idy]
        Nsct = Ysct * E * nonl
        return Nsct



    @staticmethod
    #@profile
    #@np.vectorize
    #@nb.vectorize([float64(float64, float64, float64, float64, float64)], target="parallel", nopython=True)
    @nb.vectorize([float64(float64, float64, float64, float64, float64)], target="cuda")
    def get_Ncer(E, p0, p1, p2, E0):
        """
        calculate Cherenkov photon number
        input: true deposited energy
        output: Cherenkov photon number
        """
        # p0 = gol.get_fitpar_value("p0")
        # p1 = gol.get_fitpar_value("p1")
        # p2 = gol.get_fitpar_value("p2")
        # E0 = gol.get_fitpar_value("E0")

        E = E - E0
        if E < 0:
            return 0
        else:
            #return p0 * E**2 / (E + p1 * np.exp(-p2 * E))
            return p0 * E**2 / (E + p1 * math.exp(-p2 * E))
        # return _get_Ncer(E, p0, p1, p2, E0)

    @staticmethod
    # @np.vectorize
    #@nb.vectorize([float64(float64, float64, float64, float64)], target="parallel", nopython=True)
    @nb.vectorize([float64(float64, float64, float64, float64)], target="cuda")
    def get_Nsigma(N, a, b, n):
        """
        calculate NPE sigma value
        input: NPE
        output: sigma_NPE
        """
        # a = gol.get_fitpar_value("a")
        # b = gol.get_fitpar_value("b")
        # n = gol.get_fitpar_value("n")
        # Y = gol.get_fitpar_value("Y")
        # N = E * float(Y)

        return math.sqrt(a**2 * N + b**2 * N**n)
        #return np.sqrt(a**2 * N + b**2 * N**n)
        # return _get_Nsigma(N, a, b, n)

    @staticmethod
    def _compareTruth():
        """
        Load MC truth of electrons
        return mean, sigma of NPE
        """
        mean, sigma = [], []
        for mom in range(500, 13000, 500):
            filename = f"/Volumes/home/Data/EnergyModel/electron/newsim{mom}.root"
            try:
                ff = up.open(filename)
                npe = ff["photon"]["totPE"].array()
                mean.append(np.mean(npe))
                sigma.append(np.std(npe))
            except FileNotFoundError:
                print("__ No such file : %s"%filename)
                mean.append(0)
                sigma.append(0)

        mean = np.array(mean)
        sigma = np.array(sigma)

        er_mom = np.arange(500, 13000, 500) / 1000.
        er_mu = mean
        er_sigma = sigma

        import matplotlib.pyplot as plt
        Y = gol.get_fitpar_value("Y")
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        y_nonl = (electronResponse.get_Ncer(er_mom) + electronResponse.get_Nsct(er_mom)) / Y / er_mom
        y_res = electronResponse.get_Nsigma(er_mom) / (electronResponse.get_Nsct(er_mom) + electronResponse.get_Ncer(er_mom))
        axs[0].plot(er_mom, er_mu/Y/er_mom)
        axs[0].plot(er_mom, y_nonl)
        axs[1].plot(er_mom, er_sigma/er_mu)
        axs[1].plot(er_mom, y_res)
        plt.show()
