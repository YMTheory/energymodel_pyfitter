import numpy as np
from timebudget import timebudget
import uproot as up
from scipy.stats import norm
import matplotlib.pyplot as plt

from electronResponse_ym import electronResponse
import parameters_ym as gol


class gamma(object):

    def __init__(self, name: str, E: float) -> None:
        """
        load primary electron / positron collections
        """
        self.name = name
        self.E = E

        filename = f"/Volumes/home/Data/EnergyModel/{name}_J19.root"
        print(f">>> Load Primary e+- Collections for {name}")
        ff = up.open(filename)
        gol.set_prm_value(f"{name}_elec", ff[f"{name}_elec"].to_numpy()[0])
        gol.set_prm_value(f"{name}_posi", ff[f"{name}_posi"].to_numpy()[0])
        self.elec = gol.get_prm_value(f"{name}_elec")
        self.posi = gol.get_prm_value(f"{name}_posi")

        filename = f"/Volumes/home/Data/EnergyModel/{name}_new.root"
        print(f">>> Load MC data for {name}")
        ff = up.open(filename)
        gol.set_npe_value(f"{name}", ff["photon"]["totPE"].array())
        self.npe = gol.get_npe_value(f"{name}")

        self.data_mu = np.mean(self.npe)
        self.data_sigma = np.std(self.npe)

        self.pred_mu = 0
        self.pred_sigma = 0

    def get_name(self):
        return self.name

    def get_E(self):
        return self.E

    def get_pred_mu(self):
        return self.pred_mu

    def get_pred_sigma(self):
        return self.pred_sigma



    #@timebudget
    def _calc(self):
        """
        predict mean and sigma of NPE dist.
        """
        # debugger here:
        sub_E = self.elec + self.posi
        E_per_event = np.sum(sub_E, axis=1)

        totnpe_per_event = electronResponse.get_Nsct(
            E_per_event) + electronResponse.get_Ncer(
                E_per_event) + np.count_nonzero(self.posi, axis=1) * float(
                    gol.get_fitpar_value("npeGe68"))
        pred_npe_mean = np.average(totnpe_per_event)
        #print(f"__predicted mean npe {pred_npe_mean}")

        sub_npe_sigma = electronResponse.get_Nsigma(
            self.elec)**2 + electronResponse.get_Nsigma(self.posi)**2
        sigma_per_event = np.sum(sub_npe_sigma, axis=1) + np.count_nonzero(
            self.posi, axis=1) * float(gol.get_fitpar_value("sigmaGe68"))**2
        sigma2_ave = np.average(sigma_per_event)
        sigma2_nonl = np.average((totnpe_per_event - pred_npe_mean)**2)
        pred_npe_sigma = np.sqrt(sigma2_ave + sigma2_nonl)
        #print(f"__predicted npe sigma {pred_npe_sigma}")

        return pred_npe_mean, pred_npe_sigma

    def _pdf(self, x, kB, Ysct, p0, p1, p2, E0, a, b, n):
        gol.set_fitpar_value("kB", kB)
        gol.set_fitpar_value("Ysct", Ysct)
        gol.set_fitpar_value("p0", p0)
        gol.set_fitpar_value("p1", p1)
        gol.set_fitpar_value("p2", p2)
        gol.set_fitpar_value("E0", E0)
        gol.set_fitpar_value("a", a)
        gol.set_fitpar_value("b", b)
        gol.set_fitpar_value("n", n)

        #gol._print()

        #calculate mu, sigma
        mu, sigma = self._calc()
        self.pred_mu, self.pred_sigma = mu, sigma
        #print(f"{self.name}: __predictied mu {mu}, sigma {sigma}")
        return norm.pdf(x, loc=mu, scale=sigma)


    def _print(self):
        print(f"__data: mean {self.data_mu}, sigma {self.data_sigma}")
        print(f"__calc: mean {self.pred_mu}, sigma {self.pred_sigma}")

    def _plot(self):
        Y = gol.get_fitpar_value("Y")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.hist(self.npe/Y, bins=100, density=True)
        xmin, xmax = np.min(self.npe), np.max(self.npe)
        x = np.arange(xmin, xmax, 1)
        y = norm.pdf(x, loc=self.pred_mu, scale=self.pred_sigma)
        ax.plot(x/Y, y, "-")
        ax.set_xlabel(r"$E_\mathrm{vis}$ [MeV]", fontsize=14)
        ax.set_ylabel("counts", fontsize=14)
        ax.set_title(f"{self.name}, E = {self.E} MeV", fontsize=14)
        plt.savefig(f"{self.name}.pdf")
