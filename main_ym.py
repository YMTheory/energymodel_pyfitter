from electronResponse_ym import electronResponse
import parameters_ym as gol
from gamma_source_ym import gamma
import fitter as f

if __name__ == "__main__" :

    gol._init()

    #initialize fitting parameters
    gol.set_fitpar_value_ingroup([
        "kB", "Ysct", "p0", "p1", "p2", "E0", "a", "b", "n", "Y", "npeGe68",
        "sigmaGe68"
    ], [
        5.7e-3, 1403, 92, 0.556, 0.277, 0.192, 0.974, 0.045, 1.62, 1400.0,
        1308.8, 38.067
    ])


    ### Debugger:
    er = electronResponse() ### This class must be initialized firstly !!!
  
    #Cs137 = gamma("Cs137", 0.662)
    # Cs137._pdf(1.0,
    #            kB=6.7e-3,
    #            Ysct=1413,
    #            p0=98,
    #            p1=0.556,
    #            p2=0.277,
    #            E0=0.192,
    #            a=0.974,
    #            b=0.045,
    #            n=1.62)

    f.fitter()
