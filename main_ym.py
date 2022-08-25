# import fitter as f
import parameters_ym as gol
from electronResponse_ym import electronResponse
from gamma_source_ym import gamma

if __name__ == "__main__":

    gol._init()
    gol.set_run_mode("vec")

    # initialize fitting parameters
    gol.set_fitpar_value_ingroup([
        "kB", "Ysct", "p0", "p1", "p2", "E0", "a", "b", "n", "Y", "npeGe68",
        "sigmaGe68"
    ], [
        5.76e-3, 1403.29, 91.98, 0.556, 0.277, 0.192, 0.974, 0.045, 1.62,
        1399.6, 1308.8, 38.067
    ])

    er = electronResponse()

    # er._compareTruth()

    gamma_sources = []
    gamma_names = [
        "Cs137", "Mn54", "Ge68", "K40", "nH", "Co60", "AmBe", "nC12", "AmC"
    ]
    gamma_E = [0.662, 0.835, 1.022, 1.461, 2.223, 2.506, 4.43, 4.94, 6.13]
    for name, E in zip(gamma_names, gamma_E):
        gamma_sources.append(gamma(name, E))

    for gam in gamma_sources:
        gam._calc()
        #gam._calc()
        #gam._calc()
        gam._print()
        #gam._plot()

    # f.fitter()




