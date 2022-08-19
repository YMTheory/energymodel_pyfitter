from iminuit import cost, Minuit
from timebudget import timebudget

import parameters_ym as gol
from gamma_source_ym import gamma


@timebudget
def fitter():
    gamma_sources = []
    gamma_names = ["Cs137", "Mn54", "Ge68", "K40", "nH", "Co60", "AmBe", "nC12", "AmC"]
    gamma_E = [0.662, 0.835, 1.022, 1.461, 2.223, 2.506, 4.43, 4.94, 6.13]

    for name, E in zip(gamma_names, gamma_E):
        ### For test:
        if name != "Cs137":
            continue
        gamma_sources.append(gamma(name, E))

    lf = cost.UnbinnedNLL(gol.get_npe_value(gamma_sources[0].get_name()), gamma_sources[0]._pdf)
    for i in range(len(gamma_sources)-1):
        lf += cost.UnbinnedNLL(gol.get_npe_value(gamma_sources[i+1].get_name()), gamma_sources[i+1]._pdf)
    m = Minuit(lf, kB=5.7e-3, Ysct=1400, p0=91, p1=0.5, p2=0.2, E0=0.2, a=0.98, b=0.05, n=1.62)

    m.migrad()
    m.hesse()

    print(f"Best fit {m.values}")

    for gam in gamma_sources:
        gam._print()
        gam._plot()

    return m.values, m._errors
