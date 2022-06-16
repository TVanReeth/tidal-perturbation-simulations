#!/usr/bin/env python3
#
# File: stellar_model.py
# Author: Timothy Van Reeth <timothy.vanreeth@kuleuven.be>
# License: GPL-3+
# Description: Module that builds a stellar model from a set of base parameters

import numpy as np
from astropy import units as u



class stellar_model(object):
    
    def __init__(self, Mstar, Rstar, incl_deg, frot, ld_coeff=0.6):
        """
            Reading in a stellar model from a GYRE input file (calculated with MESA),
            and setting other (custom) parameter values, including the inclination 
            angle (in degrees) and the cyclic rotation frequency (in d^{-1}) of the
            star.
            
            Parameters:
                Mstar:             astropy quantity
                                   the stellar mass
                Rstar:             astropy quantity
                                   the stellar radius 
                incl_deg:          float
                                   the inclination angle of the star in degrees
                frot:              astropy quantity
                                   the rotation frequency of the star
                ld_coeff:          float; optional
                                   limb darkening coefficient (default = 0.6)
        """
        
        # Initialsing the necessary quantities for the functions below
        # (also serves as an overview of the class variables)
        self._Mstar = Mstar
        self._Rstar = Rstar
        self._incl_deg = incl_deg
        self._frot = frot
        self._ld_coeff = ld_coeff
        self._theta = 0.
        self._phi = 0.
        self._cell_weight = 0.
        self._theta_incl = 0. 
        self._phi_incl = 0.
        
        self._set_surfacegrid()
        self._set_inclined_surface()
        
        self._pulsating = False
        
        return
    
    @property
    def Mstar(self):
        return self._Mstar
    
    @property
    def Rstar(self):
        return self._Rstar
    
    @property
    def ld_coeff(self):
        return self._ld_coeff
    
    @property
    def incl_deg(self):
        return self._incl_deg
    
    @property
    def incl(self):
        return self._incl_deg * np.pi / 180.
    
    @property
    def frot(self):
        return self._frot
    
    @property
    def theta(self):
        return self._theta
    
    @property
    def phi(self):
        return self._phi
    
    @property
    def cell_weight(self):
        return self._cell_weight
    
    @property
    def theta_incl(self):
        return self._theta_incl
    
    @property
    def phi_incl(self):
        return self._phi_incl
    
    @property
    def pulsating(self):
        return self._pulsating
    
    @Mstar.setter
    def Mstar(self, mass):
        self._Mstar = mass
    
    @Rstar.setter
    def Rstar(self, radius):
        self._Rstar = radius
    
    @incl_deg.setter
    def incl_deg(self, incl_degrees):
        self._incl_deg = incl_degrees
        self._set_inclined_surface(incl_degrees)

    @frot.setter
    def frot(self, rot_freq):
        self._frot = rot_freq
    
    @ld_coeff.setter
    def ld_coeff(self, mu):
        self._ld_coeff = mu
    
    @pulsating.setter
    def pulsating(self, puls):
        self._pulsating = puls
    
    
    def _set_surfacegrid(self, Ntheta=200, Nphi=400):
        """
            Setting the 2D-surface grid in (theta,phi) in which we 
            want to evaluate the stellar properties and pulsations,
            as well as the necessary cell_weights for when we need
            to calculate selected quantities by integrating over the
            surface.
            
            Parameters:
                self:      stellar_model object
                Ntheta:    int; optional
                           the number of grid cells in the 
                           latitudinal direction (default: 200)
                Nphi:      int; optional
                           the number of grid cells in the 
                           azimuthal direction (default: 400)
            
            Returns:
                None
        """
        
        theta = np.linspace(0, np.pi, Ntheta)
        phi = np.linspace(0., 2.*np.pi, Nphi)
        theta_weight = np.ones(len(theta))*np.pi/float(len(theta)-1)
        phi_weight = np.ones(len(phi))*np.pi/float(len(phi)-1)
        theta_weight[0] /= 2.
        theta_weight[-1] /= 2.
        phi_weight[0] /= 2.
        phi_weight[-1] /= 2.
        
        self._theta,self._phi = np.meshgrid(theta,phi)
        theta_weight,phi_weight = np.meshgrid(theta_weight,phi_weight)
        self._cell_weight = theta_weight*phi_weight*np.sin(self._theta) * 4.*np.pi / np.nansum(theta_weight*phi_weight*np.sin(self._theta))
        
        return
    
    
    
    def _set_inclined_surface(self):
        """
            Rotate the stellar model coordinates (in a spherical coordinate frame)
            over the inclination angle.

            NOTE: the rotation is always done around the y-axis. In other words, the
                angle "phi = 0" remains in the same (xz)-plane.
            
            Parameters:
                self:      stellar_model object
            
            Returns:
                None
        """
       
        x,y,z = self._spherical_to_cartesian()
        x_rot, y_rot, z_rot = self._rotate_cart(x,y,z,self.incl)
        self._theta_incl, self._phi_incl = self._cartesian_to_spher(x_rot,y_rot,z_rot)
        
        return
    
    
    
    def _spherical_to_cartesian(self):
        """
            Convert spherical surface coordinates to cartesian coordinates.
            
            Parameters:
                self:  stellar_model object

            Returns:
                x:     numpy array
                       the Cartesian coordinates on the x-axis
                y:     numpy array
                       the Cartesian coordinates on the y-axis
                z:     numpy array
                       the Cartesian coordinates on the z-axis
        """
    
        x = np.sin(self._theta) * np.cos(self._phi)
        y = np.sin(self._theta) * np.sin(self._phi)
        z = np.cos(self._theta)
        
        return x,y,z
    
    
    
    def _cartesian_to_spher(self,x,y,z):
        """
            Convert cartesian coordinates to spherical coordinates.
        
            Parameters:

                self:  stellar_model object
                x:     numpy array
                       the Cartesian coordinates on the x-axis
                y:     numpy array
                       the Cartesian coordinates on the y-axis
                z:     numpy array
                       the Cartesian coordinates on the z-axis
    
            Returns:
                theta: numpy array
                       colatitude (in rad)
                phi:   numpy array
                       azimuthal angle (in rad)
        """
        
        theta = np.arctan2(np.sqrt(x**2. + y**2.),z)
        phi = np.arctan2(y,x)
        
        return theta,phi
        
    
    
    def _rotate_cart(self,x,y,z,i_rot):
        """
            Rotate the stellar model coordinates (in a cartesian frame) over the
            angle i_rot.

            NOTE: the rotation is always done around the y-axis. In other words, the
                  angle "phi = 0" remains in the same (xz)-plane.

            Parameters:
                self:  stellar_model object
                x:     numpy array
                       the original coordinates on the x-axis
                y:     numpy array
                       the original coordinates on the y-axis
                z:     numpy array
                       the original coordinates on the z-axis
                i_rot: float
                       the angle over which we rotate the model (in rad).
     
            Returns:
                x_rot: numpy array
                       the rotated coordinates on the x-axis
                y_rot: numpy array
                       the rotated coordinates on the y-axis
                z_rot: numpy array
                       the rotated coordinates on the z-axis
    
        """
        
        x_rot = np.cos(i_rot)*x - np.sin(i_rot)*z
        y_rot = y
        z_rot = np.sin(i_rot)*x + np.cos(i_rot)*z
    
        return x_rot, y_rot, z_rot
    
    
    
    def limb_darkening(self):
        """
            Basic linear limb darkening law, applied to the observed stellar surface.
    
            Parameters:
                self:      stellar_model object
    
            Returns:
                ld:        numpy array
                           relative intensity of the observed stellar disk
        """
    
        ld = 1. - self._ld_coeff * (1. - np.cos(self.theta_incl))
        return ld
    
        

    def rotational_velocityfield(self):
        """
            Calculate the velocity field of the star in the inertial reference
            frame, caused by the stellar rotation
    
            NOTE 1: the rotation is always done around the y-axis. In other words,
                    the angle "phi = 0" remains in the same (xz)-plane.
        
            Returns:
                radv:  astropy Quantity array
                       the radial component of the rotational velocity field in the
                       inertial reference frame (in km/s)
                thv:   astropy Quantity array
                       the colatitudinal component of the rotational velocity field
                       in the inertial reference frame (in km/s)
                phv:   astropy Quantity array
                       the azimuthal component of the rotational velocity field in
                       the inertial reference frame (in km/s)
        """
        
        radv = np.zeros(self._theta.shape) * u.km / u.s
        thv =  np.zeros(self._theta.shape) * u.km / u.s
        phv = 2. * np.pi * self._Rstar.to(u.km) * np.sin(self._theta) * self.frot.to(u.Hz)
    
        return radv, thv, phv
        
        
    
