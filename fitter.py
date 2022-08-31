from iminuit import cost, Minuit
from timebudget import timebudget

import parameters_ym as gol
from gamma_source_ym import gamma
from BetaSpectrum import BetaSpectrum, GenericLeastSquares, BetterLeastSquares

@timebudget
def fitter():
    #############################
    ####### gamma sources #######
    #############################
    gamma_sources = []
    gamma_names = ["Cs137", "Mn54", "Ge68", "K40", "nH", "Co60", "AmBe", "nC12", "AmC"]
    gamma_E = [0.662, 0.835, 1.022, 1.461, 2.223, 2.506, 4.43, 4.94, 6.13]

    for name, E in zip(gamma_names, gamma_E):
        ### For test:
        # if name != "Cs137":
        #     continue
        gamma_sources.append(gamma(name, E))

    nll = cost.UnbinnedNLL(gol.get_npe_value(gamma_sources[0].get_name()), gamma_sources[0]._pdf)
    for i in range(len(gamma_sources)-1):
        nll += cost.UnbinnedNLL(gol.get_npe_value(gamma_sources[i+1].get_name()), gamma_sources[i+1]._pdf)


    #############################
    ########### B12 #############
    #############################gg
    b12 = BetaSpectrum("B12", 1000, 0, 15, 80, 3, 12)
    b12._load_data()
    b12._load_theo()
    dataX = b12.get_dataX()
    dataY = b12.get_dataY()
    dataYe = b12.get_dataYe()
    ## lf += cost.LeastSquares(dataX, dataY, dataYe, b12._pdf)
    lsq = BetterLeastSquares(b12._pdf, dataX, dataY, dataYe)

    csum = nll + lsq

    m = Minuit(csum, kB=5.7e-3, Ysct=1400, p0=91, p1=0.5, p2=0.2, E0=0.2, a=0.98, b=0.05, n=1.62)
    # setting parameters ranges
    m.limits["kB"] = (5.0e-3, 9.0e-3)
    m.limits["Ysct"] = (1350, 1450)
    m.limits["p0"] = (50, 150)
    m.limits["p1"] = (0, 1)
    m.limits["p2"] = (0, 1)
    m.limits["E0"] = (0, 0.3)
    m.limits["a"] = (0.8, 1.2)
    m.limits["b"] = (0, 0.5)
    m.limits["n"] = (1.0, 2.0)

    m.migrad()
    m.hesse()

    print("======== Fitting Outputs ========")
    print(f"Best fit {m.values}")
    print(f"Fit errors {m.errors}")
    print(m.covariance)
    print(m.fmin)
    print("=================================")

    for gam in gamma_sources:
        gam._print()
        gam._plot()

    return m.values, m._errors
