from electronResponse_cpu import electronResponse
import parameters_ym as glb

import uproot as up
import boost_histogram as bh

class BetaSpectrum:

    def __init__(self, name, nE, Emin, Emax, nEvis, Evismin, Evismax):
        self.name    = name
        self.dfile   = f"/Volumes/home/Data/EnergyModel/{name}_data.root"
        self.tfile   = f"/Volumes/home/Data/EnergyModel/{name}_theo.root"
        self.Emin    = Emin
        self.Emax    = Emax
        self.nE      = nE
        self.Evismin = Evismin
        self.Evismax = Evismax
        self.nEvis   = nEvis

        self.dhist = bh.Histogram(bh.axis.Regular(nEvis, Evismin, Evismax))
        self.thist = bh.Histogram(bh.axis.Regular(nE, Emin, Emax))
    

    def _load_theo(self):
        try:
           f = up.open(self.tfile)
           arr = f[self.name]["edep"].array()
           self.thist.fill(arr)
        except FileExistsError:
            print(f"The {self.name} theoretical file does not exist, the fitting procedure can not be executed furthermore...:(")
            raise FileExistsError
        


    def _load_data(self):
        try:
            f = up.open(self.dfile)
            arr = f[self.name]["totpe"].array()
            self.dhist.fill(arr/glb.get_fitpar_value("Y"))

        except FileExistsError:
            print(f"The {self.name} data file does not exist, the fitting procedure can not be executed furthermore...:(")
            raise FileExistsError
        
