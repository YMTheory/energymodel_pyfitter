import numpy as np
import numba as nb
import parameters as gol

        
class gamma_source(object):

    #def __init__(self, name:str, E:float )->None:
    def __init__( self, E:float )->None:
        #self.name = name
        self.E = E
        self.secondaries_elec = np.zeros((5000, 100))
        self.secondaries_posi = np.zeros((5000, 100))

        
    @nb.njit
    def calculator(self)->None:
        _calculator(self.secondaries_elec, self.secondaries_posi)






gol._init()
Cs137 = gamma_source(0.662)
Cs137.calculator()

