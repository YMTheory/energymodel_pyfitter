from electronResponse_ym import electronResponse
import parameters_ym as gol
from gamma_source_ym import gamma

if __name__ == "__main__" :

    gol._init()

    ## initialize fitting parameters
    gol.set_fitpar_value_ingroup(["kB", "Ysct", "p0", "p1", "p2", "E0", "a", "b", "n", "Y", "npeGe68", "sigmaGe68"], 
                                 [5.7e-3, 1403, 92, 0.556, 0.277, 0.192, 0.974, 0.045, 1.62, 1400.0, 1308.8, 38.067]
                                 )
    

    er = electronResponse()

    Cs137 = gamma("Cs137", 0.662)
    Cs137._pdf(1.0, [5.7e-3, 1403, 92, 0.556, 0.277, 0.192, 0.974, 0.045, 1.62])
    ### Debugger:
    #print(f"Scintillation #: {er.get_Nsct(1)}, Cherenkov #: {er.get_Ncer(1)}")


