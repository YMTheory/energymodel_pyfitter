import numpy as np
from iminuit import cost, Minuit
import uproot as up
import numba as nb
from numba import vectorize, float64
from timebudget import timebudget
import time
from scipy.stats import norm

kBmin, kBmax    = 45e-4, 95e-4
quenchNL_E      = np.arange(5e-4, 14.9995, 1e-3)
quenchNL        = np.zeros((50, 14999))
quenchNL_file   = "/Volumes/home/Data/EnergyModel/Quench_NumInt.root"

#electron meters :
Ysct    = 1403.29
kB      = 5.76e-3
kC      = 0.98
p0      = 91.98
p1      = 0.556
p2      = 0.277
E0      = 0.192
a       = 0.974
b       = 0.045
n       = 1.62


# detector configurations
Y           = 3111.44 / 2.223
Ge68mean    = 1.30880e3
Ge68sigma   = 38.0672


# gamma data
nSources = 9
names = ["Cs137", "Mn54", "Ge68", "K40", "nH", "Co60", "AmBe", "nC12", "AmC"]
Etrue = [0.662, 0.835, 1.022, 1.461, 2.223, 2.506, 4.43, 4.94, 6.13]

npe_data = np.zeros((nSources, 50000))
secondaries_elec = np.zeros((nSources, 5000, 100))
secondaries_posi = np.zeros((nSources, 5000, 100))

pred_npe_mean = np.zeros(nSources)
pred_npe_sigma = np.zeros(nSources)


def initialize()->None:
    """
    load quenching nonlinearity file: load into array
    load gamma NPE data: load into2 array
    load gamma primary beta: load into 3*3 array
    """

    print("------ Initialzed electron response -----")
    fquenchNL = up.open(quenchNL_file)
    print("Quenching nonlinearity is loaded from %s"%quenchNL_file)
    for ikb in range(45, 95, 1):
        hname               = "kB" + str(ikb)
        hquenchNL           = fquenchNL[f"{hname}"]
        quenchNL[ikb-45]    = hquenchNL.to_numpy()[0]


    for n, gamma_name in enumerate(names):
        filename = f"/Volumes/home/Data/EnergyModel/{gamma_name}_new.root"
        ff = up.open(filename)
        print("Load MC data from %s ..."%filename, end=", ")
        npe_data[n] = ff["photon"]["totPE"].array()

        filename = f"/Volumes/home/Data/EnergyModel/{gamma_name}_J19.root"
        ff = up.open(filename)
        print("primary beta from %s ..."%filename)
        secondaries_elec[n] = ff[f"{gamma_name}_elec"].to_numpy()[0]
        secondaries_posi[n] = ff[f"{gamma_name}_posi"].to_numpy()[0]


@vectorize([float64(float64, float64, float64)])
def get_Nsct_vec(E, Ysct, kB):
    """ return scintilation photon numbers, input Edep"""
    ## linear interpolation seems take a lot time in the calculation, requires more optimization...
    #kBlow  = int((kB - kBmin)*10000)
    #kBhigh = kBlow + 1 
    #r = ((kB - kBmin )*10000 - kBlow) / (kBhigh - kBlow)
    #quenchNL_low  = np.interp(E, quenchNL_E, quenchNL[kBlow])
    #quenchNL_high = np.interp(E, quenchNL_E, quenchNL[kBhigh])
    #nl = r * quenchNL_high + (1-r)*quenchNL_low
    idx, idy = int((kB - kBmin)*10000), int(E/0.001)
    nl =  quenchNL[idx, idy]
    N = nl * Ysct * E
    return N

@vectorize([float64(float64, float64, float64, float64)])
def get_Ncer_vec(E, p0, p1, p2):
    """ return Cherenkov photon numbers, input Edep"""
    return p0 * E**2 / (E + p1 * np.exp(-p2 * E))


@timebudget
@nb.njit(parallel=True)
def prediction_oneSample(Y, Ysct, kB, p0, p1, p2, E0, a, b, n):
    #elecC = np.where(secondaries_elec[])
    pass

    
@timebudget
@nb.njit(parallel=True)
def prediction(Y, Ysct, kB, a, b, n, p0, p1, p2, E0):
    elecC = np.where(secondaries_elec-E0>0, secondaries_elec-E0, 0)
    mean_NPE_elec = get_Nsct_vec(secondaries_elec, Ysct, kB) + get_Ncer_vec(elecC, p0, p1, p2)
    sigma_NPE_elec  = get_Nsigma_vec(secondaries_elec * Y, a, b, n)
    posiC = np.where(secondaries_posi-E0>0, secondaries_posi-E0, 0)
    mean_NPE_posi = get_Nsct_vec(secondaries_posi, Ysct, kB) + get_Ncer_vec(posiC, p0, p1, p2)
    sigma_NPE_posi  = get_Nsigma_vec(secondaries_posi * Y, a, b, n)

    ## calculate mean values
    elec1 = np.sum(mean_NPE_elec, axis=2)
    posi1 = np.sum(mean_NPE_posi, axis=2)
    posi2 = posi1 + Ge68mean * np.count_nonzero(mean_NPE_posi, axis=2)
    gam_mean = elec1 + posi2
    pred_npe_mean = np.sum(elec1+posi2, axis=1) / (elec1+posi2).shape[1]
    pred_npe_mean2D = np.zeros(gam_mean.shape)
    for i in range(gam_mean.shape[0]):
        for j in range(gam_mean.shape[1]):
            pred_npe_mean2D[i, j] = pred_npe_mean[i]

    ### calculation sigma values
    elec3 = np.sum(sigma_NPE_elec**2, axis=2)
    posi3 = np.sum(sigma_NPE_posi**2, axis=2)
    posi4 = posi3 + Ge68sigma**2 * np.count_nonzero(sigma_NPE_posi, axis=2)
    sigma2_ave = np.sum(elec3+posi4, axis=1) / (elec3+posi4).shape[1]
    tmp = (gam_mean - pred_npe_mean2D)**2
    sigma2_nonl = np.sum(tmp, axis=1) / tmp.shape[1]
    pred_npe_sigma = np.sqrt( sigma2_nonl + sigma2_ave )
    return pred_npe_mean, pred_npe_sigma


def get_pred_mean():
    return pred_npe_mean


def get_pred_sigma():
    return pred_npe_sigma


@vectorize([float64(float64, float64, float64, float64)])
def get_Nsigma_vec(N, a, b, n):
    return np.sqrt(a**2*N + b**2* N**n)


def pdf(x, Y, Ysct, kB, a, b, n, p0, p1, p2, E0):
    pdf_arr = []
    mu, sigma = prediction(Y, Ysct, kB, a, b, n, p0, p1, p2, E0)
    for i, j in zip(mu, sigma):
        pdf_arr.append(norm(x, i, y))

    return pdf_arr

def fit(x):
    mu, sigma = prediction(Y, Ysct, kB, a, b, n, p0, p1, p2, E0)
    pdf_arr = pdf()

    lf = cost.UnbinnedNLL(data, pdf)


    

def main():
    initialize()
    x = np.arange(0, 10000, 1)
    fit(x)


if __name__ == "__main__" :
    main()















