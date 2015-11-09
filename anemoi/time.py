from .discretization import BaseDiscretization

import numpy as np
import scipy.sparse

class Time(BaseDiscretization):

    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'nTsamp':       (False,     '_nTsamp',        np.int64),
        'f_max':        (True,      None,           np.complex128),
        'source_freq':  (True,      None,           np.complex128),
        'tau':          (False,     '_tau',         np.float64),
    }

    def _Forward(self):
    """
    Takes the time parameters that you are interested in and gives you suggested frequency
    range.
    """

    nx = self.nx
    nz = self.nz
    dims = (nz, nx)
    nrows = nx*nz

    dx = self.dx
    dz = self.dz

    #determine the maximum length of the Model
    x_max = (nx-1)*dx
    z_max = (nz-1)*dz
    L_max = np.max([xmax,zmax])

    c = self.c
    c_min = self.c.min()

    #calculate the smallest wavelength, in meters
    lambda_min = c_min/f_max

    #determine the maximum modelled time from the maximum length of the grid
    #and the minimum velocity

    t_max = 2*(L_max/c_min)

    #time-domain damping factor

    f_damp = self.tau
    tau = f_damp*t_max

    #determine the frequency interval from the maximum modelled time

    df= 1/t_max

    #determine the number of frequencies
    f_max = self.f_max
    nf = f_max/df

    #determine the sampling interval and number of time domain samples

    dt = 1 / (2*f_max)
    nt = t_max/dt

    f_min = f_max - ((nf-1)*df)

    freq_range=np.arange(f_min,f_max+df,df)

    return freq_range

    def _Keuper(self):
    """
    Takes the time paramters that you are interested in and generates a time-domain
    keuper wavelet.
    """

    nt = self.nt
    freq_range=self._freq_range
    dt = self.dt
    source_freq = self.source_freq
    excursions=1.

    def Get_K_Wavelet(source_freq,dt,excursions)

    m = (excursions+2)/excursions
    nsrc = (1./source_freq)/dt
    delta = excursions * np.pi * source_freq
    loop_vals = np.arange(1,nsrc+1,1)

    for i in range(1,nsrc+1):

        tsrc(i) = (i-1.)*dt
        source(i,1) = (delta * np.cos(delta*tsrc(i))) - (delta*np.cos(m*delta*tsrc(i)))
    return source

    temp_source = Get_K_Wavelet(source_freq,dt,excursions)
    time_source = np.zeros(nt,1)
    time_source[:len(temp_source)] = temp_source[:]

    return time_source

    def _Source_FFT(self):
    """
    Takes the time domain wavelet and converts to frequency domain
    """

    time_source = self.time_source
    nt = self.nt
    nTsamp = self.nTsamp
    freq_source=np.fft.fft(time_source,[nTsamp])
    freq_source[:(nt/2)+1]=0

    return freq_source


    def _IFFT(self):
    """
    Takes the frquency domain pressure field and converts to the time domain
    pressure field. Data input has dimensions(f,z,x) where f is the modelled frequency
    corresponding to each z by x array of pressure values.
    """

    freqs = self.freq_range
    df = self.df
    nTsamp = self.nTsamp

    u = u.reshape(u.shape[0],-1)

    u_t = np.fft.ifft(u,[nTsamp])
    u_t = u_t.reshape(-1,nz,nx)

    return u_t

@property
def freq_range(self):
    if getattr(self, '_freq_range', None) is None:
        self._freq_range = self._Forward()
    return self._freq_range

@property
def time_source(self):
    if getattr(self, '_time_source', None) is None:
        self._time_source = self._Keuper()
    return self._time_source

@property
def freq_source(self):
    if getattr(self, '_freq_source', None) is None:
        self._freq_source = self._Source_FFT()
    return self._freq_source

@property
def u_t(self):
    if getattr(self, '_u_t', None) is None:
        self._u_t = self._IFFT()
    return self._u_t
