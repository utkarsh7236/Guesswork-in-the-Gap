"""a module that houses basic cosmology logic
- Modified by Utkarsh Mali
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------
import numpy as np
#-------------------------------------------------

### Planck 2018 Cosmology (Table1 in arXiv:1807.06209)
PLANCK_2018_Ho = 2.1816926176539463e-18 ### CGS
PLANCK_2018_OmegaMatter = 0.3158
PLANCK_2018_OmegaLambda = 1. - PLANCK_2018_OmegaMatter
PLANCK_2018_OmegaRadiation = 0.
# PLANCK_2018_OmegaKappa = 0.

### define units in SI
C_SI =  299792458.0
PC_SI = 3.085677581491367e+16
MPC_SI = PC_SI * 1e6
G_SI = 6.6743e-11
MSUN_SI = 1.9884099021470415e+30

### define units in CGS
G_CGS = G_SI * 1e+3
C_CGS = C_SI * 1e2
PC_CGS = PC_SI * 1e2
MPC_CGS = MPC_SI * 1e2
MSUN_CGS = MSUN_SI * 1e3

#-------------------------------------------------

DEFAULT_DZ = 1e-3 ### should be good enough for most numeric integrations we want to do

#-------------------------------------------------

class Cosmology(object):
    """\
a class that implements specific cosmological computations.
**NOTE**, we work in CGS units throughout, so Ho must be specified in s**-1 and distances are specified in cm
    """

    def __init__(self, Ho, OmegaMatter, OmegaRadiation, OmegaLambda):
        self._Ho = Ho
        self._OmegaMatter = OmegaMatter
        self._OmegaRadiation = OmegaRadiation
        self._OmegaLambda = OmegaLambda

        assert self.OmegaKappa==0, 'we only implement flat cosmologies! OmegaKappa must be 0'

        self._init_memo() ### instantiate the memorized interpolation arrays

    #---

    @property
    def Ho(self):
        return self._Ho

    @property
    def c_over_Ho(self):
        C_SI = 299792458.0
        C_CGS = C_SI * 1e2
        return C_CGS/self.Ho

    #---

    @property
    def OmegaMatter(self):
        return self._OmegaMatter

    @property
    def OmegaRadiation(self):
        return self._OmegaRadiation

    @property
    def OmegaLambda(self):
        return self._OmegaLambda

    @property
    def OmegaKappa(self):
        return 1. - (self.OmegaMatter + self.OmegaRadiation + self.OmegaLambda)

    #---

    @property
    def distances(self):
        return self._distances ### instantiated within _init_memo

    @property
    def z(self):
        return self.distances['z']

    @property
    def DL(self):
        return self.distances['DL']

    @property
    def Dc(self):
        return self.distances['Dc']

    @property
    def Vc(self):
        return self.distances['Vc']

    #---

    def _init_memo(self):
        """instantiate things to "memorize" results of distance calculations, which we later interpolate
        """
        self._distances = {
            'z':np.array([0]),
            'DL':np.array([0]),
            'Dc':np.array([0]),
            'Vc':np.array([0]),
        }

    def extend(self, max_DL=-np.inf, max_Dc=-np.inf, max_z=-np.inf, max_Vc=-np.inf, dz=DEFAULT_DZ):
        """integrate to solve for distance measures.
        """
        ### note, this could be slow due to trapazoidal approximation with small step size

        # extract current state
        distances = self.distances

        z_list = list(self.z)
        Dc_list = list(self.Dc)
        Vc_list = list(self.Vc)

        current_z = z_list[-1]
        current_Dc = Dc_list[-1]
        current_DL = current_Dc * (1+current_z)
        current_Vc = Vc_list[-1]

        # initialize integration
        current_dDcdz = self.dDcdz(current_z)
        current_dVcdz = self.dVcdz(current_z, current_Dc)

        # iterate until we are far enough
        while (current_Dc < max_Dc) or (current_DL < max_DL) or (current_z < max_z) or (current_Vc < max_Vc):
            current_z += dz                                ### increment

            dDcdz = self.dDcdz(current_z)                  ### evaluated at the next step
            current_Dc += 0.5*(current_dDcdz + dDcdz) * dz ### trapazoidal approximation
            current_dDcdz = dDcdz                          ### update

            dVcdz = self.dVcdz(current_z, current_Dc)      ### evaluated at the next step
            current_Vc += 0.5*(current_dVcdz + dVcdz) * dz ### trapazoidal approximation
            current_dVcdz = dVcdz                          ### update

            current_DL = (1+current_z)*current_Dc          ### update

            Dc_list.append(current_Dc)                     ### append
            Vc_list.append(current_Vc)
            z_list.append(current_z)

        # record
        self._distances['z'] = np.array(z_list, dtype=float)
        self._distances['Dc'] = np.array(Dc_list, dtype=float)
        self._distances['Vc'] = np.array(Vc_list, dtype=float)
        self._distances['DL'] = (1. + self.z)*self.Dc ### only holds in a flat universe

    #---

    def z2E(self, z):
        """returns E(z) = sqrt(OmegaLambda + OmegaKappa*(1+z)**2 + OmegaMatter*(1+z)**3 + OmegaRadiation*(1+z)**4)
        """
        one_plus_z = 1. + z
        return (self.OmegaLambda + self.OmegaKappa*one_plus_z**2 + self.OmegaMatter*one_plus_z**3 + self.OmegaRadiation*one_plus_z**4)**0.5

    def dDcdz(self, z):
        """returns (c/Ho)/E(z)
        """
        return self.c_over_Ho/self.z2E(z)

    def dDLdz(self, z, dz=DEFAULT_DZ):
        """returns Dc + (1+z)*dDcdz
        """
        return self.z2Dc(z, dz=dz) + (1+z)*self.dDcdz(z)

    def dVcdz(self, z, Dc=None, dz=DEFAULT_DZ):
        """returns dVc/dz
        """
        if Dc is None:
            Dc = self.z2Dc(z, dz=dz)
        return 4*np.pi * Dc**2 * self.dDcdz(z)

    def logdVcdz(self, z, Dc=None, dz=DEFAULT_DZ):
        """return ln(dVc/dz), useful when constructing probability distributions without overflow errors
        """
        if Dc is None:
            Dc = self.z2Dc(z, dz=dz)
        return np.log(4*np.pi) + 2*np.log(Dc) + np.log(self.dDcdz(z))

    #---

    def Dc2z(self, Dc, dz=DEFAULT_DZ):
        """return redshifts for each Dc specified.
        """
        max_Dc = np.max(Dc)
        if max_Dc > np.max(self.Dc):
            self.extend(max_Dc=max_Dc, dz=dz)
        return np.interp(Dc, self.Dc, self.z)

    def z2Dc(self, z, dz=DEFAULT_DZ):
        """return Dc for each z specified
        """
        max_z = np.max(z)
        if max_z > np.max(self.z):
            self.extend(max_z=max_z, dz=dz)
        return np.interp(z, self.z, self.Dc)

    #---

    def DL2z(self, DL, dz=DEFAULT_DZ):
        """returns redshifts for each DL specified.
        """
        max_DL = np.max(DL)
        if max_DL > np.max(self.DL): ### need to extend the integration
            self.extend(max_DL=max_DL, dz=dz)
        return np.interp(DL, self.DL, self.z)

    def z2DL(self, z, dz=DEFAULT_DZ):
        """returns luminosity distance at the specified redshifts
        """
        max_z = np.max(z)
        if max_z > np.max(self.z):
            self.extend(max_z=max_z, dz=dz)
        return np.interp(z, self.z, self.DL)

    #---

    def Vc2z(self, Vc, dz=DEFAULT_DZ):
        max_Vc = np.max(Vc)
        if max_Vc > np.max(self.Vc):
            self.extend(max_Vc=max_Vc, dz=DEFAULT_DZ)

        return np.interp(Vc, self.Vc, self.z)

    def z2Vc(self, z, dz=DEFAULT_DZ):
        max_z = np.max(z)
        if max_z > np.max(self.z):
            self.extend(max_z=max_z, dz=DEFAULT_DZ)

        return np.interp(z, self.z, self.Vc)

    def z2dVcdz(self, z, dz=DEFAULT_DZ):
        return self.dVcdz(z, dz=DEFAULT_DZ)

#-------------------------------------------------

# define default cosmology

PLANCK_2018_Cosmology = Cosmology(
    PLANCK_2018_Ho,
    PLANCK_2018_OmegaMatter,
    PLANCK_2018_OmegaRadiation,
    PLANCK_2018_OmegaLambda,
)
