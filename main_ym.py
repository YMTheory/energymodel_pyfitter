# import fitter as f
import parameters_ym as gol
import parameters_cuda as gol_cuda
# from electronResponse_cuda import electronResponse
# from electronResponse_cpu import electronResponse
from electronResponse_plain import electronResponse
from gamma_source_ym import gamma
from BetaSpectrum import BetaSpectrum

if __name__ == "__main__":

    # initialize fitting parameters
    gol.set_fitpar_value_ingroup([
        "kB", "Ysct", "p0", "p1", "p2", "E0", "a", "b", "n", "Y", "npeGe68",
        "sigmaGe68"
    ], [
        5.76e-3, 1403.29, 91.98, 0.556, 0.277, 0.192, 0.974, 0.045, 1.62,
        1399.6, 1308.8, 38.067
    ])

    electronResponse()
    gol.set_run_mode("cpu")
    if gol.get_run_mode() == "cuda":
        gol_cuda.copy_quenchNL_to_device()

    """
    gamma_sources = []
    gamma_names = [
        "Cs137", "Mn54", "Ge68", "K40", "nH", "Co60", "AmBe", "nC12", "AmC"
    ]
    gamma_E = [0.662, 0.835, 1.022, 1.461, 2.223, 2.506, 4.43, 4.94, 6.13]
    for name, E in zip(gamma_names, gamma_E):
        #if name != "Cs137":
        #    continue
        gamma_sources.append(gamma(name, E))

    for gam in gamma_sources:
        # gam._calc()
        # gam._calc()
        gam._calc()
        gam._print()
        #gam._plot()

    # f.fitter()
    """

    b12 = BetaSpectrum("B12", 1000, 0, 15, 80, 3, 12)
    b12._load_theo()
    b12._load_data()
    # b12.ApplyResponse()
    b12.ApplyResponse_cpu() 

    # #b12._plot()
