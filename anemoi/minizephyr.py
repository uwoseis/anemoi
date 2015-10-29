
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

# Configuration defaults for MiniZephyr
DTYPE_COMPLEX = np.complex128
DTYPE_REAL = np.float64

class MiniZephyr(object):

    c           =   None
    rho         =   None
    nPML        =   None
    freq        =   None
    ky          =   None
    dx          =   None
    dx          =   None
    nx          =   None
    nz          =   None
    freeSurf    =   None

    def __init__(self, systemConfig):

        initMap = {
        #   Argument        Rename to Property
            'c':            '_c',
            'rho':          '_rho',
            'nPML':         '_nPML',
            'freq':         None,
            'ky':           '_ky',
            'dx':           '_dx',
            'dz':           '_dz',
            'nx':           None,
            'nz':           None,
            'freeSurf':     '_freeSurf',
            'mord':         '_mord',
            'premul':       '_premul',
        }

        for key in initMap.keys():
            if key in systemConfig:
                if initMap[key] is None:
                    setattr(self, key, systemConfig[key])
                else:
                    setattr(self, initMap[key], systemConfig[key])

    def _initHelmholtzNinePoint(self):
        """
        An attempt to reproduce the finite-difference stencil and the
        general behaviour of OMEGA by Pratt et al. The stencil is a 9-point
        second-order version based on work by a number of people in the mid-90s
        including Ivan Stekl. The boundary conditions are based on the PML
        implementation by Steve Roecker in fdfdpml.f.
        """

        dtypeComplex = DTYPE_COMPLEX
        dtypeReal = DTYPE_REAL

        nx = self.nx
        nz = self.nz
        dims = (nz, nx)
        nrows = nx*nz

        c = self.c
        rho = self.rho

        exec 'nf = %s'%self.mord[0] in locals()
        exec 'ns = %s'%self.mord[1] in locals()

        # fast --> slow is x --> y --> z as Fortran

        # Set up physical properties in matrices with padding
        omega   = 2*np.pi * self.freq
        cPad    = np.pad(c, pad_width=1, mode='edge')
        rhoPad  = np.pad(rho, pad_width=1, mode='edge')

        aky = 2*np.pi * self.ky

        # Horizontal, vertical and diagonal geometry terms
        dx  = self.dx
        dz  = self.dz
        freeSurf = self.freeSurf

        dxx = dx**2
        dzz = dz**2
        dxz = (dxx+dzz)/2
        dd  = np.sqrt(dxz)
        iom = 1j * omega

        # PML decay terms
        # NB: Arrays are padded later, but 'c' in these lines
        #     comes from the original (un-padded) version

        nPML    = self.nPML

        pmldx   = dx*(nPML - 1)
        pmldz   = dz*(nPML - 1)
        pmlr    = 1e-3
        pmlfx   = 3.0 * np.log(1/pmlr)/(2*pmldx**3)
        pmlfz   = 3.0 * np.log(1/pmlr)/(2*pmldz**3)

        dpmlx   = np.zeros(dims, dtype=dtypeComplex)
        dpmlz   = np.zeros(dims, dtype=dtypeComplex)
        isnx    = np.zeros(dims, dtype=dtypeReal)
        isnz    = np.zeros(dims, dtype=dtypeReal)

        # Only enable PML if the free surface isn't set

        if not freeSurf[2]:
            isnz[-nPML:,:] = -1 # Top

        if not freeSurf[1]:
            isnx[:,-nPML:] = -1 # Right Side

        if not freeSurf[0]:
            isnz[:nPML,:] = 1 # Bottom

        if not freeSurf[3]:
            isnx[:,:nPML] = 1 # Left side

        dpmlx[:,:nPML] = (np.arange(nPML, 0, -1)*dx).reshape((1,nPML))
        dpmlx[:,-nPML:] = (np.arange(1, nPML+1, 1)*dx).reshape((1,nPML))
        dnx     = pmlfx*c*dpmlx**2
        ddnx    = 2*pmlfx*c*dpmlx
        denx    = dnx + iom 
        r1x     = iom / denx
        r1xsq   = r1x**2
        r2x     = isnx*r1xsq*ddnx/denx

        dpmlz[:nPML,:] = (np.arange(nPML, 0, -1)*dz).reshape((nPML,1))
        dpmlz[-nPML:,:] = (np.arange(1, nPML+1, 1)*dz).reshape((nPML,1))
        dnz     = pmlfz*c*dpmlz**2
        ddnz    = 2*pmlfz*c*dpmlz
        denz    = dnz + iom
        r1z     = iom / denz
        r1zsq   = r1z**2
        r2z     = isnz*r1zsq*ddnz/denz

        # Visual key for finite-difference terms
        # (per Pratt and Worthington, 1990)
        #
        #   This         Original
        # AF FF CF  vs.  AD DD CD
        # AA BE CC  vs.  AA BE CC
        # AD DD CD  vs.  AF FF CF

        # Set of keys to index the dictionaries
        keys = ['AD', 'DD', 'CD', 'AA', 'BE', 'CC', 'AF', 'FF', 'CF']

        # Diagonal offsets for the sparse matrix formation
        offsets = {
            'AD':   -nf -ns,
            'DD':   -nf    ,
            'CD':   -nf +ns,
            'AA':       -ns,
            'BE':         0,
            'CC':       +ns,
            'AF':   +nf -ns,
            'FF':   +nf    ,
            'CF':   +nf +ns,
        }

        def prepareDiagonals(diagonals):
            for key in diagonals:
                diagonals[key] = diagonals[key].ravel()
                if offsets[key] < 0:
                    diagonals[key] = diagonals[key][-offsets[key]:]
                elif offsets[key] > 0:
                    diagonals[key] = diagonals[key][:-offsets[key]]
                diagonals[key] = diagonals[key].ravel()

        # Buoyancies
        bMM = 1. / rhoPad[0:-2,0:-2] # bottom left
        bME = 1. / rhoPad[0:-2,1:-1] # bottom centre
        bMP = 1. / rhoPad[0:-2,2:  ] # bottom right
        bEM = 1. / rhoPad[1:-1,0:-2] # middle left
        bEE = 1. / rhoPad[1:-1,1:-1] # middle centre
        bEP = 1. / rhoPad[1:-1,2:  ] # middle right
        bPM = 1. / rhoPad[2:  ,0:-2] # top    left
        bPE = 1. / rhoPad[2:  ,1:-1] # top    centre
        bPP = 1. / rhoPad[2:  ,2:  ] # top    right

        # Initialize averaged buoyancies on most of the grid
        bMM = (bEE + bMM) / 2 # a2
        bME = (bEE + bME) / 2 # d1
        bMP = (bEE + bMP) / 2 # d2
        bEM = (bEE + bEM) / 2 # a1
        # ... middle
        bEP = (bEE + bEP) / 2 # c1
        bPM = (bEE + bPM) / 2 # f2
        bPE = (bEE + bPE) / 2 # f1
        bPP = (bEE + bPP) / 2 # c2

        # Model parameter M
        K = ((omega**2 / cPad**2) - aky**2) / rhoPad

        # K = omega^2/(c^2 . rho)
        kMM = K[0:-2,0:-2] # bottom left
        kME = K[0:-2,1:-1] # bottom centre
        kMP = K[0:-2,2:  ] # bottom centre
        kEM = K[1:-1,0:-2] # middle left
        kEE = K[1:-1,1:-1] # middle centre
        kEP = K[1:-1,2:  ] # middle right
        kPM = K[2:  ,0:-2] # top    left
        kPE = K[2:  ,1:-1] # top    centre
        kPP = K[2:  ,2:  ] # top    right

        # 9-point fd star
        acoef   = 0.5461
        bcoef   = 0.4539
        ccoef   = 0.6248
        dcoef   = 0.09381
        ecoef   = 0.000001297

        # 5-point fd star
        # acoef = 1.0
        # bcoef = 0.0
        # ecoef = 0.0

        # NB: bPM and bMP here are switched relative to S. Roecker's version
        #     in OMEGA. This is because the labelling herein is always ?ZX.

        diagonals = {
            'AD':   ecoef*kMM
                    + bcoef*bMM*((r1zsq+r1xsq)/(4*dxz) - (r2z+r2x)/(4*dd)),
            'DD':   dcoef*kME
                    + acoef*bME*(r1zsq/dz - r2z/2)/dz
                    + bcoef*(r1zsq-r1xsq)*(bMP+bMM)/(4*dxz),
            'CD':   ecoef*kMP
                    + bcoef*bMP*((r1zsq+r1xsq)/(4*dxz) - (r2z-r2x)/(4*dd)),
            'AA':   dcoef*kEM
                    + acoef*bEM*(r1xsq/dx - r2x/2)/dx
                    + bcoef*(r1xsq-r1zsq)*(bPM+bMM)/(4*dxz),
            'BE':   ccoef*kEE
                    + acoef*(r2x*(bEM-bEP)/(2*dx) + r2z*(bME-bPE)/(2*dz) - r1xsq*(bEM+bEP)/dxx - r1zsq*(bME+bPE)/dzz)
                    + bcoef*(((r2x+r2z)*(bMM-bPP) + (r2z-r2x)*(bMP-bPM))/(4*dd) - (r1xsq+r1zsq)*(bMM+bPP+bPM+bMP)/(4*dxz)),
            'CC':   dcoef*kEP
                    + acoef*bEP*(r1xsq/dx + r2x/2)/dx
                    + bcoef*(r1xsq-r1zsq)*(bMP+bPP)/(4*dxz),
            'AF':   ecoef*kPM
                    + bcoef*bPM*((r1zsq+r1xsq)/(4*dxz) + (r2z-r2x)/(4*dd)),
            'FF':   dcoef*kPE
                    + acoef*bPE*(r1zsq/dz + r2z/2)/dz
                    + bcoef*(r1zsq-r1xsq)*(bPM+bPP)/(4*dxz),
            'CF':   ecoef*kPP
                    + bcoef*bPP*((r1zsq+r1xsq)/(4*dxz) + (r2z+r2x)/(4*dd)),
        }

        self._setupBoundary(diagonals, freeSurf)

        prepareDiagonals(diagonals)

        diagonals = [diagonals[key] for key in keys]
        offsets = [offsets[key] for key in keys]

        A = scipy.sparse.diags(diagonals, offsets, shape=(nrows, nrows), format='csr', dtype=dtypeComplex)

        return A

    def _setupBoundary(self, diagonals, freeSurf):
        """
        Function to set up boundary regions for the Seismic FDFD problem
        using the 9-point finite-difference stencil from OMEGA/FULLWV.
        """

        keys = diagonals.keys()
        pickDiag = lambda x: -1. if freeSurf[x] else 1.

        # Left
        for key in keys:
            if key is 'BE':
                diagonals[key][:,0] = pickDiag(3)
            else:
                diagonals[key][:,0] = 0.

        # Right
        for key in keys:
            if key is 'BE':
                diagonals[key][:,-1] = pickDiag(1)
            else:
                diagonals[key][:,-1] = 0.

        # Bottom
        for key in keys:
            if key is 'BE':
                diagonals[key][0,:] = pickDiag(0)
            else:
                diagonals[key][0,:] = 0.

        # Top
        for key in keys:
            if key is 'BE':
                diagonals[key][-1,:] = pickDiag(2)
            else:
                diagonals[key][-1,:] = 0.

    @property
    def A(self):
        if getattr(self, '_A', None) is None:
            self._A = self._initHelmholtzNinePoint()
        return self._A

    @property
    def Solver(self):
        if getattr(self, '_Solver', None) is None:
            A = self.A.tocsc()
            self._Solver = scipy.sparse.linalg.splu(A)
        return self._Solver

    @property
    def mord(self):
        return getattr(self, '_mord', ('+nx', '+1'))
    
    @property
    def c(self):
        if isinstance(self._c, np.ndarray):
            return self._c
        else:
            return self._c * np.ones((self.nz, self.nx), dtype=DTYPE_COMPLEX)

    @property
    def rho(self):
        if getattr(self, '_rho', None) is None:
            self._rho = 310. * self.c**0.25 
            
        if isinstance(self._rho, np.ndarray):
            return self._rho
        else:
            return self._rho * np.ones((self.nz, self.nx), dtype=DTYPE_REAL)

    @property
    def nPML(self):
        return getattr(self, '_nPML', 10)

    @property
    def ky(self):
        return getattr(self, '_ky', 0.)

    @property
    def dx(self):
        return getattr(self, '_dx', 1.)
    
    @property
    def dz(self):
        return getattr(self, '_dz', self.dx)
    
    @property
    def freeSurf(self):
        if getattr(self, '_freeSurf', None) is None:
            self._freeSurf = (False, False, False, False)
        return self._freeSurf
    
    @property
    def premul(self):
        return getattr(self, '_premul', 1.)

    def __mul__(self, value):
        u = self.premul * self.Solver.solve(value)
        return u
