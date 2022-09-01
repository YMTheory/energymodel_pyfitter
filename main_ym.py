import parameters_ym as gol
import parameters_cuda as gol_cuda
# from electronResponse_cuda import electronResponse
from electronResponse_cpu import electronResponse
# from electronResponse_plain import electronResponse
import fitter as f

import sys

if __name__ == "__main__":

    for i in range(len(sys.argv)):
        arg = sys.argv[i]
        if arg == "-server_name":
            gol.set_server_name(sys.argv[i+1])
            print(f"Current server name: {sys.argv[i+1]}")

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

    gol.set_fit_gam_flag(True)
    gol.set_fit_gam_method("binned")
    gol.set_fit_B12_flag(False)

    f.fitter()
