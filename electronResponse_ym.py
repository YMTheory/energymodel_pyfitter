import numpy as np
import parameters_ym as gol
import logging
import uproot as up
import numba as nb

class electronResponse(object):

    def __init__(self) -> None:
        """
        load quenching nonlinearity curve clusters into _global_kB_dict
        """
        print("------ Initialzed electron response -----")
        quenchNL_file   = "/Volumes/home/Data/EnergyModel/Quench_NumInt.root"
        fquenchNL = up.open(quenchNL_file)
        logging.debug("Quenching nonlinearity is loaded from %s"%quenchNL_file)

        Emin, Emax, Estep = 5e-4, 14.9995, 1e-3
        gol.set_quenchE(np.arange(Emin, Emax, Estep))

        kBmin, kBmax = 45, 95
        for ikb in range(kBmin, kBmax, 1):
            hname               = "kB" + str(ikb)
            try:
                hquenchNL           = fquenchNL[f"{hname}"]
                gol.set_kB_value(hname, hquenchNL.to_numpy()[0])
            except:
                gol.set_kB_value(hname, np.zeros(len(gol.get_quenchE())))
    

    @staticmethod
    @np.vectorize
    def get_Nsct(E):
        """
        calculate scintillation photon number
        input: true deposited energy
        output: scintillation photon number
        """
        kB      = gol.get_fitpar_value("kB")
        Ysct    = gol.get_fitpar_value("Ysct")
        
        idx, idy = int((kB)*10000), int(E/0.001)
        nl = gol.get_kB_value(f"kB{idx}")[idy]
        N = nl * Ysct * E
        return N


    @staticmethod
    @np.vectorize
    def get_Ncer(E):
        """
        calculate Cherenkov photon number
        input: true deposited energy
        output: Cherenkov photon number
        """
        p0  = gol.get_fitpar_value("p0")
        p1  = gol.get_fitpar_value("p1")
        p2  = gol.get_fitpar_value("p2")
        E0  = gol.get_fitpar_value("E0")

        E = E - E0
        if E0 < 0:
            return 0
        else:
            return p0*E**2 / (E + p1*np.exp(-p2*E))


    @staticmethod
    def get_Nsigma(E):
        """
        calculate NPE sigma value
        input: true deposited energy
        output: sigma_NPE
        """
        a   = gol.get_fitpar_value("a")
        b   = gol.get_fitpar_value("b")
        n   = gol.get_fitpar_value("n")
        Y   = gol.get_fitpar_value("Y")
        N   = E * float(Y)
        
        return np.sqrt(a**2 * N + b**2 * N**n)








