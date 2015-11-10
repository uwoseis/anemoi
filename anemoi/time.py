from .meta import AttributeMapper
import numpy as np

class BaseTime(AttributeMapper):

    initMap = {
        #   Argument        Required    Rename as ...   Store as type
        'nTSamp':       (False,     '_nTSamp',        np.int64),
        'fMax':        (True,      None,           np.complex128),
        'sourceFreq':  (True,      None,           np.complex128),
        'tau':          (False,     '_tau',         np.float64),
        'dx':           (False,     '_dx',          np.float64),
        'dz':           (False,     '_dz',          np.float64),
        'nx':           (True,      None,           np.int64),
        'nz':           (True,      None,           np.int64),
    }

    @property
    def dx(self):
        return getattr(self, '_dx', 1.)

    @property
    def dz(self):
        return getattr(self, '_dz', 1.)

    @property
    def nTSamp(self):
        return getattr(self, '_nTSamp', 256)

    @property
    def tau(self):
        return getattr(self, '_tau', 0.4)

class Forward(BaseTime):
    """
    Takes the time parameters that you are interested in and gives you suggested frequency
    range.
    """
    def __init__(self, systemConfig):

        super(Forward, self).__init__(systemConfig)

        fMax = self.fMax
        nf = self.nf
        df = self.df

    def __call__(self):

        fMin = fMax - ((nf - 1) * df)

        freqRange = np.arange(fMin,fMax + self.df,self.df)

        return freqRange

class Keuper(BaseTime):
    """
    Takes the time paramters that you are interested in and generates a time-domain
    keuper wavelet.
    """
    def __init__(self, systemConfig):

        super(Keuper, self).__init__(systemConfig)

        nt = self.nt
        dt = self.dt
        sourceFreq = self.sourceFreq
        excursions=1.

    def __call__(self):

        def getKWavelet(sourceFreq,dt,excursions):

            m = (excursions + 2) / excursions
            nsrc = (1. / sourceFreq) / dt
            delta = excursions * np.pi * sourceFreq
            loopVals = np.arange(1,nsrc + 1,1)

            for i in range(1,nsrc+1):

                tsrc[i-1] = (i - 1.) * dt
                source[i-1] = (delta * np.cos(delta *tsrc[i-1])) - (delta *np.cos(m * delta *tsrc[i-1]))
            return source

        tempSource = getKWavelet(sourceFreq,dt,excursions)
        timeSource = np.zeros(nt,1)
        timeSource[:len(tempSource)] = tempSource[:]

        return timeSource

class sourceFFT(self):
        """
        Takes the time domain wavelet and converts to frequency domain
        """
    def __init__(self, systemConfig):

        super(sourceFFT, self).__init__(systemConfig)

        nt = self.nt
        dt = self.dt
        nTSamp = self.nTSamp

    def __call__(self, timeSource):

        freqSource=np.fft.fft(timeSource,[nTSamp])
        freqSource[:(nt / 2) + 1]=0

        return freqSource


class iFFT(self):
        """
        Takes the frquency domain pressure field and converts to the time domain
        pressure field. Data input has dimensions(f,z,x) where f is the modelled frequency
        corresponding to each z by x array of pressure values.
        """
    def __init__(self, systemConfig):

        super(iFFT, self).__init__(systemConfig)

        nTSamp = self.nTSamp
        nx = self.nx
        nz = self.nz

    def __call__(self, u):

        uF = u.reshape(u.shape[0],-1)
        uT = np.fft.ifft(uF,[nTSamp])
        uT = uT.reshape(-1,nz,nx)

        return uT
