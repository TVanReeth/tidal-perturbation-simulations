#!/usr/bin/env python3
#
# File: gmode_pulsation.py
# Author: Timothy Van Reeth <timothy.vanreeth@kuleuven.be>
# License: GPL-3+
# Description: Module describing the basic properties of a g-mode pulsation
#              with the Traditional Approximation of Rotation (TAR)

import numpy as np
import sys
from astropy import units as u

from Hough import hough_functions


class gmode_pulsation(object):
    
    def __init__(self, gyre_dir, puls_freq, puls_ampl, frot, k, m):
        """
            Loading in the TAR module from GYRE v5.x or v6.x, and reading in
            additional parameters describing the studied g-mode.
            
            Parameters:
                gyre_dir:           string
                                    The GYRE 5.x or 6.x installation directory.
                puls_freq:          astropy quantity
                                    (observed) cyclic pulsation frequency in the 
                                    inertial reference frame
                puls_ampl:          float
                                    peak amplitude of the pulsation on the stellar 
                                    surface (unit = mmag)
                frot:               astropy quantity
                                    cyclic rotation frequency of the studied star
                k:                  int
                                    latitudinal degree of the g-mode
                m:                  int
                                    azimuthal order of the g-mode
        """
        
        
        self._puls_freq_co = np.abs(puls_freq - m*frot)
        self._puls_freq = self._puls_freq_co + m*frot
        self._amplitude = puls_ampl
        self._k = k
        self._m = m
        self._l = np.abs(k) + np.abs(m)
        self._lam_fun = self._retrieve_lambda(gyre_dir, self._k, self._m)
        
        self._spin = float(2. * frot / self._puls_freq_co)
        self._lmdb = self._lam_fun(self._spin)
        self._Hr = 0.
        self._Ht = 0.
        self._Hp = 0.
        
        return
    
    
    @property
    def puls_freq(self):
        return self._puls_freq
    
    @property
    def puls_freq_co(self):
        return self._puls_freq_co
    
    @property
    def amplitude(self):
        return self._amplitude
    
    @property
    def k(self):
        return self._k
    
    @property
    def l(self):
        return self._l
    
    @property
    def m(self):
        return self._m
    
    @property
    def Hr(self):
        return self._Hr
    
    @property
    def Ht(self):
        return self._Ht
    
    @property
    def Hp(self):
        return self._Hp
    
    
    def lam(self, spin):
        """
            Calculate the eigenvalue Lambda of the Laplace Tidal Equation
            that corresponds to the given spin value for the mode identi-
            fication (k,m) of the studied g-mode.
            
            Parameters:
                self:   gmode_pulsation object
                spin:   float
                        the spin parameter value s = 2*nu_rot/nu_co, where
                        nu_rot is the stellar rotation frequency and nu_co
                        is the pulsation frequency in the corotating frame.
            
            Returns:
                lmbda:  float
                        the eigenvalue lambda corresponding to the given
                        spin
        """
        
        lmbda = self._lam_fun(spin) 
        
        return lmbda       
        
        
    
    def _retrieve_lambda(self, gyre_dir, kval, mval):
        """
            Retrieving the function lambda(nu) given in GYRE v5.x or v6.x.
    
            Parameters:
                self:     gmode_pulsation object
                gyre_dir: string
                          The GYRE 5.x or 6.x installation directory.
                kval:     int/float
                          latitudinal degree of the g-mode
                mval:     int/float
                          azimuthal order of the g-mode
    
            Returns:
                lam_fun:  function
                          A function to calculate lambda, given spin parameter
                          values as input.
        """

        if(kval >= 0):
            kstr = f'+{kval}'
        else:
            kstr = f'{kval}'
        if(mval >= 0):
            mstr = f'+{mval}'
        else:
            mstr = f'{mval}'
    
        infile = f'{gyre_dir}/data/tar/tar_fit.m{mstr}.k{kstr}.h5'
    
        sys.path.append(gyre_dir+'/src/tar/') 
        import gyre_tar_fit
        import gyre_cheb_fit
    
        tf = gyre_tar_fit.TarFit.load(infile)
        lam_fun = np.vectorize(tf.lam)
    
        return lam_fun
    
    
    
    def calculate_puls_geometry(self, star):
        """
            A wrapper around the Hough function calculation by Vincent Prat, to get
            the 2D geometry of the g-mode pulsation on the surface of this star.
    
            Parameters:
                self:       gmode_pulsation object
                star:       stellar_model object
                            our studied stellar model

            Computes:
                self._Hr:   numpy array
                            the radial component of the Hough function (as a
                            function of theta)
                self._Ht:   numpy array
                            the latitudinal component of the Hough function (as a
                            function of theta)
                self._Hp:   numpy array
                            the azimuthal component of the Hough function (as a
                            function of theta)
        """
        
        (lmbd_min, mu, hr, ht, hp) = hough_functions(self._spin, self._l, self._m, lmbd=-self._lmdb,npts=1000)
    
        hr_mapped = np.ones(star.theta.shape)
        ht_mapped = np.ones(star.theta.shape)
        hp_mapped = np.ones(star.theta.shape)
    
        for irow in np.arange(len(star.theta[:,0])):
            hr_mapped[irow][::-1] = np.interp(np.cos(star.theta[irow,::-1]),mu[::-1],hr[::-1])
            ht_mapped[irow][::-1] = np.interp(np.cos(star.theta[irow,::-1]),mu[::-1],ht[::-1])
            hp_mapped[irow][::-1] = np.interp(np.cos(star.theta[irow,::-1]),mu[::-1],hp[::-1])
        
        self._Hr = hr_mapped
        self._Ht = ht_mapped
        self._Hp = hp_mapped
        
        return


    
    
