#!/usr/bin/env python3
#
# File: plot_tidal_perturbation_model.py
# Author: Timothy Van Reeth <timothy.vanreeth@kuleuven.be>
# License: GPL-3+
# Description: Routines to visualize the simulations calculated with
#              the tidal_perturbation_in_circular_binary.py script

import glob
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

mpl.rc('font',size=14)
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'


def linearfit(time, flux, freqs):
     """
         A (fast) linear fit of sine waves to a light curve
         
         Parameters:
             time:      numpy array (dtype = float)
                        the time stamps of the flux observations
             flux:      numpy array (dtype = float)
                        the measured flux values that are evaluated
             freqs:     numpy array (dtype = float)
                        the frequencies of the sine waves that are evaluated
         Returns:
             amp:       float
                        the amplitude of the fitted sine wave
             phase:     numpy array (dtype = float)
                        the phase of the fitted sine wave, with values between
                        -0.5 and 0.5.
     """
 
     matrix = []
     for freq in freqs:
         matrix.append(np.sin(2.*np.pi*freq*time))
         matrix.append(np.cos(2.*np.pi*freq*time))
     matrix = np.array(matrix).T
 
     try:
         par = np.linalg.inv(matrix.T @ matrix) @ matrix.T @ flux
         par1 = np.array([par[2*ii] for ii in np.array(np.linspace(0,len(freqs)-1,len(freqs)),dtype=int)])
         par2 = np.array([par[2*ii+1] for ii in np.array(np.linspace(0,len(freqs)-1,len(freqs)),dtype=int)])
         amp = np.sqrt(par1**2. + par2**2.)
         phase = np.arctan2(par2,par1) / (2.*np.pi)
         
     except:
         print("Warning: (likely) singular matrix in the lineatfit() routine")
         amp = np.zeros(len(freqs))
         phase = np.zeros(len(freqs))
     
     return amp[0], phase[0]



def plot_theoretical_perturbations(filename, bin_ax, amp_ax, ph_ax, clr, label=None):
    """
        Routine to plot the toy model g-mode perturbations, assuming that the studied star
        is perfectly synchronously rotating with the binary orbit.
        
        Parameters:
            filename:     string
                          the absolute path to the filename with the calculated model
            bin_ax:       matplotlib axes object
                          axes where the phase-folded light curve is plotted
            amp_ax:       matplotlib axes object
                          axes where the amplitude modulations are plotted
            ph_ax:        matplotlib axes object
                          axes where the phase modulations are plotted
            clr:          matplotlib color
                          the color in which the data have to be shown
            label:        string; optional
                          the label of the data that have to be shown. (default: None)
    """
    
    dat = np.genfromtxt(filename, names=True)
    orb_phases = np.unique(dat['orb_phase'])
    orb_phases.sort()
    orb_phases_ext = np.array(list(orb_phases-1) + list(orb_phases) + list(orb_phases+1))
    
    freqval = float(filename.split('_')[-3][1:]) / 1000000.
    
    vis_amp = []
    vis_ph = []
    for ival in orb_phases:
        sel = np.r_[dat['orb_phase'] == ival]
        ind = np.argsort(dat['puls_phase'][sel])
        aval,phval = linearfit(dat['time'][sel],dat['pulsation'][sel]-np.nanmean(dat['pulsation'][sel]),np.array([freqval]))
        vis_amp.append(aval)
        vis_ph.append(phval)
    vis_amp = np.array(vis_amp)
    vis_ph = np.array(vis_ph)
    
    if(label is None):
        bin_ax.plot(np.array(list(dat['orb_phase']-1) + list(dat['orb_phase']) + list(dat['orb_phase']+1)),np.array(3*list(dat['total_flux'])),'k.',c=clr,markersize=2)
        amp_ax.plot(orb_phases_ext,np.array(3*list(vis_amp/np.nanmean(vis_amp))),'k-',c=clr)
        ph_ax.plot(orb_phases_ext,np.array(3*list(vis_ph-np.nanmean(vis_ph))),'k-',c=clr) 
    else:
        bin_ax.plot(orb_phases_ext,np.array(3*list(dat['total_flux'])),'k.',c=clr,markersize=2, label=label)
        amp_ax.plot(orb_phases_ext,np.array(3*list(vis_amp/np.nanmean(vis_amp))),'k-',c=clr, label=label)
        ph_ax.plot(orb_phases_ext,np.array(3*list(vis_ph-np.nanmean(vis_ph))),'k-',c=clr)
    
    return


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
        define a subsection of an existing colormap as a new colormap
        
        Parameters:
            cmap:      matplotlib colormap
                       the input (full) colormap
            minval:    float; optional
                       the minimum value, marking the start of the new colormap
                       (values between 0 and 1; default = 0).
            maxval:    float; optional
                       the maximum value, marking the end of the new colormap
                       (values between 0 and 1; default = 1).
            n:         integer; optional
                       the number of colors selected to be the new colormap
                       (default = 100)
        
        Returns:
            new_cmap:  matplotlib colormap
                       the output (subset) colormap
    """
    
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
    
    return new_cmap



if __name__ == "__main__":
    
    maindir = '/lhome/timothyv/LiGang_binaries/scripts/Theoretical_toy-model/'
    
    # For this demo I am looping over simulated pulsations with different frequencies, but in the same system
    model_filenames = glob.glob(f'{maindir}binary_M150M140_forb1000_i70/gmode_f*_perturbed-visibilities_???.dat')
    model_filenames.sort()
    
    cmap = truncate_colormap(mpl.cm.get_cmap('hot'), minval=0., maxval=0.8)
    
    fig = plt.figure(1,figsize=(6,6))
    plt.subplots_adjust(top=0.97,left=0.17,right=0.95,hspace=0.15,bottom=0.10)
    bin_ax = fig.add_subplot(311)
    plt.xticks([-0.2,0.,0.2,0.4,0.6,0.8,1.,1.2],['','','','','','','',''])
    plt.ylabel('flux (mmag)')
    
    amp_ax = fig.add_subplot(312)
    plt.xticks([-0.2,0.,0.2,0.4,0.6,0.8,1.,1.2],['','','','','','','',''])
    plt.ylabel('A / <A>')
    
    ph_ax = fig.add_subplot(313)
    plt.xlabel('orbital phase')
    plt.ylabel(r'$\sf \phi - <\phi>$ (2$\pi$ rad)')
    
    for ifile,filename in enumerate(model_filenames):
        plot_theoretical_perturbations(filename, bin_ax, amp_ax, ph_ax, cmap(float(ifile)/len(model_filenames)))
    
    bin_ax.xaxis.set_minor_locator(AutoMinorLocator())
    amp_ax.xaxis.set_minor_locator(AutoMinorLocator())
    ph_ax.xaxis.set_minor_locator(AutoMinorLocator())
    bin_ax.yaxis.set_minor_locator(AutoMinorLocator())
    amp_ax.yaxis.set_minor_locator(AutoMinorLocator())
    ph_ax.yaxis.set_minor_locator(AutoMinorLocator())
    bin_ax.yaxis.set_ticks_position('both')
    bin_ax.xaxis.set_ticks_position('both')
    amp_ax.yaxis.set_ticks_position('both')
    amp_ax.xaxis.set_ticks_position('both')
    ph_ax.yaxis.set_ticks_position('both')
    ph_ax.xaxis.set_ticks_position('both')
    
    bin_ax.set_xlim(-0.2,1.2)
    amp_ax.set_xlim(-0.2,1.2)
    ph_ax.set_xlim(-0.2,1.2)
    
    bin_ax.invert_yaxis()
    
    fig.align_ylabels([bin_ax,amp_ax,ph_ax])
    plt.show()












