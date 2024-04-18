import numpy as np

def normSw(fSw,fSwirr,fSnrw):
    return (fSw-fSwirr)/(1.0-fSnrw-fSwirr)

'''Water relative permeability'''
class coreyWater:
    def __init__(self,Nw,Krwo,Swirr,Sorw):
        '''
        Args:
        Nw: Exponent
        Krwo: Relperm at 1-S_{orw}
        Swirr: S_{wi}
        Sorw: S_{orw}
        '''
        self.Nw = Nw
        self.Krwo = Krwo
        self.Swirr = Swirr
        self.Sorw = Sorw
    def __call__(self,Sw):
        nSw = normSw(Sw,self.Swirr,self.Sorw)
        value = self.Krwo*nSw**self.Nw
        value = np.maximum(1E-8,value)
        return value
        #return self.Krwo*nSw**self.Nw
    
class coreyOil:
    def __init__(self,No,Swirr,Sorw):
        '''
        Args:
        No: Exponent
        Swirr: S_{wi}
        Sorw: S_{orw}
        '''
        self.No = No
        self.Swirr = Swirr
        self.Sorw = Sorw
    def __call__(self,Sw):
        nSw = normSw(Sw,self.Swirr,self.Sorw)
        value = (1.0-nSw)**self.No
        value = np.maximum(1E-8,value)
        return value
        #return (1.0-nSw)**self.No