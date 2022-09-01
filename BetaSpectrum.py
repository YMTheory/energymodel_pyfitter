#from electronResponse_cuda import electronResponse
from electronResponse_cpu import electronResponse
# from electronResponse_plain import electronResponse
import parameters_ym as glb

import uproot as up
import boost_histogram as bh
import numpy as np
from scipy.stats import norm
from timebudget import timebudget
import numba as nb
import math


class BetaSpectrum:

    def __init__(self, name, nE, Emin, Emax, nEvis, Evismin, Evismax):
        self.name = name
        # self.dfile = f"../data/{name}_data.root"
        # self.tfile = f"../data/{name}_theo.root"
        self.dfile = f"/Volumes/home/Data/EnergyModel/{name}_data.root"
        self.tfile = f"/Volumes/home/Data/EnergyModel/{name}_theo.root"
        self.Emin = Emin
        self.Emax = Emax
        self.nE = nE
        self.EbinWidth = (Emax - Emin) / nE
        self.Evismin = Evismin
        self.Evismax = Evismax
        self.nEvis = nEvis
        self.EvisbinWidth = (Evismax - Evismin) / nEvis

        self.dhist = bh.Histogram(bh.axis.Regular(nEvis, Evismin, Evismax))
        self.thist = bh.Histogram(bh.axis.Regular(nE, Emin, Emax))
        self.phist = bh.Histogram(bh.axis.Regular(nEvis, Evismin, Evismax))

        self.m_bin_center = np.zeros(self.nEvis)
        self.m_data_content = np.zeros(self.nEvis)
        self.m_pred_content = np.zeros(self.nEvis)
        self.m_data = None

    def _load_theo(self):
        try:
            f = up.open(self.tfile)
            arr = f[self.name]["edep"].array()
            self.thist.fill(arr)
            print(f">>> Load {self.name} theoretical file.")
        except FileExistsError:
            print(
                f"The {self.name} theoretical file does not exist, the fitting procedure can not be executed furthermore...:("
            )
            raise FileExistsError

    def _load_data(self):
        try:
            f = up.open(self.dfile)
            arr = f[self.name]["totpe"].array()
            self.m_data = arr
            self.dhist.fill(arr / glb.get_fitpar_value("Y"))
            self.m_bin_center = self.dhist.axes[0].centers
            self.m_data_content = self.dhist.view()
            print(f">>> Load {self.name} data file.")

        except FileExistsError:
            print(
                f"The {self.name} data file does not exist, the fitting procedure can not be executed furthermore...:("
            )
            raise FileExistsError

    def get_dataX(self):
        return self.m_bin_center

    def get_dataY(self):
        return self.m_data_content

    def get_dataYe(self):
        return np.sqrt(self.m_data_content)

    def get_full_data(self):
        return self.m_data

    @timebudget
    def ApplyResponse(self):

        Ysct = glb.get_fitpar_value("Ysct")
        p0 = glb.get_fitpar_value("p0")
        p1 = glb.get_fitpar_value("p1")
        p2 = glb.get_fitpar_value("p2")
        E0 = glb.get_fitpar_value("E0")
        a = glb.get_fitpar_value("a")
        b = glb.get_fitpar_value("b")
        n = glb.get_fitpar_value("n")
        Y = glb.get_fitpar_value("Y")

        electronResponse.update()
        snonl = electronResponse.get_nonl()

        self.m_pred_content = np.zeros(self.nEvis)
        m_eTru = self.thist.view()

        for i in range(self.nE):
            eTrue = self.thist.axes[0].centers[i]
            eTrueID = int(eTrue * 1000)
            tmp_npe = electronResponse.get_Nsct(
                eTrue, eTrueID, snonl, Ysct) + electronResponse.get_Ncer(
                    eTrue, p0, p1, p2, E0)
            tmp_sigma = electronResponse.get_Nsigma(tmp_npe, a, b, n)

            ### Smearing with energy resolution
            minEbin = int(((tmp_npe - 5 * tmp_sigma) / Y - self.Evismin) /
                          self.EvisbinWidth)
            maxEbin = int(((tmp_npe + 5 * tmp_sigma) / Y - self.Evismin) /
                          self.EvisbinWidth)
            for ilocbin in range(minEbin, maxEbin + 1):
                if ilocbin < 0 or ilocbin > self.nEvis:
                    continue
                tmp_E = self.Evismin + (ilocbin + 0.5) * self.EvisbinWidth
                prob = norm.pdf(tmp_E, loc=tmp_npe / Y, scale=tmp_sigma)
                self.m_pred_content[ilocbin] += prob * m_eTru[i]

    def _plot(self):
        import matplotlib.pyplot as plt
        for i in range(self.nEvis):
            self.phist[i] = self.m_pred_content[i]

        fig, ax = plt.subplots()
        ax.plot(self.dhist.axes[0].centers,
                self.dhist.view(),
                "o",
                ms=4,
                color="red",
                label="Simulation")
        ax.plot(self.phist.axes[0].centers,
                self.phist.view(),
                "-",
                lw=2,
                color="black",
                label="Best fit")
        ax.set_xlabel(r"$E_\mathrm{vis}$ [MeV]", fontsize=14)
        ax.set_ylabel("count", fontsize=14)
        ax.legend(prop={"size": 14})
        plt.savefig(f"./figures/{self.name}.pdf")

    #@timebudget
    def ApplyResponse_cpu(self):
        Ysct = glb.get_fitpar_value("Ysct")
        p0 = glb.get_fitpar_value("p0")
        p1 = glb.get_fitpar_value("p1")
        p2 = glb.get_fitpar_value("p2")
        E0 = glb.get_fitpar_value("E0")
        a = glb.get_fitpar_value("a")
        b = glb.get_fitpar_value("b")
        n = glb.get_fitpar_value("n")
        Y = glb.get_fitpar_value("Y")

        electronResponse.update()
        snonl = electronResponse.get_nonl()

        # fine binning:
        m_cont = self.thist.view()  ## content
        m_cent = self.thist.axes[0].centers
        idE = (m_cent * 1000).astype("int")
        Nsct = np.zeros_like(m_cent)
        m_npe = electronResponse.get_Nsct(m_cent, idE, snonl, Ysct,
                                          Nsct) + electronResponse.get_Ncer(
                                              m_cent, p0, p1, p2, E0)
        m_sigma = electronResponse.get_Nsigma(m_npe, a, b, n)

        self.m_pred_content = BetaSpectrum.smear(Y, self.Evismin,
                                                 self.EvisbinWidth, self.nEvis,
                                                 m_npe, m_sigma, m_cont)

        ## Normalization:
        sum_data = np.sum(self.m_data_content)
        sum_pred = np.sum(self.m_pred_content)
        self.m_pred_content = self.m_pred_content / sum_pred * sum_data

    #@timebudget
    @staticmethod
    #@nb.njit
    def smear(Y, Evismin, EvisbinWidth, nEvis, m_npe, m_sigma, m_theo_content):
        PI = 3.141592653
        m_cont = np.zeros(nEvis)
        for i in range(len(m_npe)):
            tmp_npe = m_npe[i]
            tmp_sigma = m_sigma[i]
            minEbin = int(
                ((tmp_npe - 5 * tmp_sigma) / Y - Evismin) / EvisbinWidth)
            maxEbin = int(
                ((tmp_npe + 5 * tmp_sigma) / Y - Evismin) / EvisbinWidth)
            for ilocbin in range(minEbin, maxEbin + 1):
                if ilocbin < 0 or ilocbin >= nEvis:
                    continue
                tmp_E = Evismin + (ilocbin + 0.5) * EvisbinWidth
                #prob = norm.pdf(tmp_E, loc=tmp_npe / Y, scale=tmp_sigma)
                prob = 1 / (math.sqrt(2 * PI) * tmp_sigma) * math.exp(
                    -(tmp_E - tmp_npe / Y)**2 / 2 / tmp_sigma**2)
                m_cont[ilocbin] += prob * m_theo_content[i]
        return m_cont

    def _pdf(self, x, kB, Ysct, p0, p1, p2, E0, a, b, n):
        """ Gaussian PDF for NPE distribution """
        glb.set_fitpar_value("kB", kB)
        glb.set_fitpar_value("Ysct", Ysct)
        glb.set_fitpar_value("p0", p0)
        glb.set_fitpar_value("p1", p1)
        glb.set_fitpar_value("p2", p2)
        glb.set_fitpar_value("E0", E0)
        glb.set_fitpar_value("a", a)
        glb.set_fitpar_value("b", b)
        glb.set_fitpar_value("n", n)

        self.ApplyResponse_cpu()

        return np.interp(x, self.m_bin_center, self.m_pred_content)

    ###def _chi2(self):
    ###    self.ApplyResponse_cpu()

    ###    m_data = self.dhist.view()
    ###    m_pred = self.phist.view()
    ###    masked = np.ma.masked_values(m_data, 0).data
    ###    mask_data = masked.data
    ###    mask = masked.mask
    ###    mask_pred = np.ma.masked_array(m_pred, mask)

    ###    chi2 = np.sum((mask_pred - mask_data)**2 / mask_data)
    ###    return chi2


###from iminuit import Minuit
###from iminuit.util import describe
###from iminuit.util import make_func_code
###
###class GenericLeastSquares:
###    """
###    Generic least-square cost function with errors.
###    """
###    errordef = Minuit.LEAST_SQUARES
###
###    def __init__(self, model, x, y, yerr):
###        self.model = model  # model predicts y for given x
###        self.x = np.asarray(x)
###        self.y = np.asarray(y)
###        self.err = np.asarray(yerr)
###
###    def __call__(self, *par):
###        ym = np.zeros(len(self.y))
###        for ix in range(len(self.x)):
###            ym[ix] = self.model(self.x[ix], *par)
###        return np.sum((self.y - ym) ** 2 / self.err ** 2)
###
###class BetterLeastSquares(GenericLeastSquares):
###    def __init__(self, model, x, y, yerr):
###        super().__init__(model, x, y, yerr)
###        self.func_code = make_func_code(describe(model)[1:])
