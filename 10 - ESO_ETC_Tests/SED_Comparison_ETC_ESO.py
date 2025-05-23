# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 16:25:44 2025

@author: MatteoGenoni
"""

import math #ok
import numpy as np #ok
from numpy import exp, cos, linspace #ok
import matplotlib.pyplot as plt #ok
import os, time, glob
import scipy.special as sp
from ftplib import FTP
import gzip
from scipy import interpolate, constants, sqrt, exp
from astropy.io import fits
from os import listdir # CANCELLARE
from os.path import isfile, join
from operator import truediv
from glob import glob
from astropy.table import Table


import tools
from tools import zp_norm
from tools import sed as sd
from tools import mag as mg
from tools import converter


# Sec 0 : Input parameter definition
# Sec 0.1
sed = 1 # [0:BB , 1:Power-Law, 3:input spectrum, 4:Single Emission Line]
sed_t = 3000 # Temperature in [K]
sed_index = 2 # Power-Law index [-]
mag = 7
magsystem = 1  # [1:VEGA , 2:AB]
bandm = 1  # From below
bandm_list = ["U", "B", "V", "R", "I", "J", "H", "K"]
z = 0  # Redshift
passo = 0.1 # [A]
# Sec 0.2 : DEBUG mode
DEBUG = True
unit = 0  # 0 = ergs ; 1 = ph
bp = True  # bandpass
# Sec 0.3 : SED loading from database
cwd = os.getcwd()
folder_load = '\ESPRESSO'
folder_load_ins = (cwd + folder_load)
filename_load = '\PowerLaw_2__B_7_VV__z0.txt'
filepath_load = (folder_load_ins + filename_load)


# Sec 1 : Filter and Mag-System
# Sec 1.1 Filter
lambda_0, zeropoint = mg.get_vega_flux_zeropoints(bandm_list[bandm], "PHll")
# sez 1.2 MagSystem --> always working in the Vega System
magsys = 'Vega'
if magsystem == 2:
	magsys = 'AB'
	# Convert the MAGNITUDE from AB to VEGA (Default Mag System used in the code)
	mag = mg.ab_to_vega_converter(mag, bandm_list[bandm])

if DEBUG:
	print("MAG (Vega): " + str(mag))
	print("ll zero: " + str(lambda_0))
	print("Zeropoint: " + str(zeropoint))
	

# Sec 2 : Input Science Object SED
# Sec 2.1 : Wavelength Range in [A]
wave_sorgente = np.arange(3000.0, 9000.0, passo / (1. + z))
# wave_sorgente = np.round([200 + (x * (passo / (1. + z))) for x in range(0, int((26000.1 - 200) * (1 / (passo / (1. + z)))))], 2)

# Sec 2.2 : SED cases
x_cut = None
# 0: BLACKBODY
if sed == 0:
	wave_ricevuta = (1. + z) * wave_sorgente
	flusso_energia = sd.get_blackbody(wave_sorgente, sed_t)  # J/(s * m2 * m)
	sed_name = "Blackbody"
	'''
	if DEBUG:
		plots.generic(wave_sorgente, [[flusso_energia, sed_name]], sed_name +" SED", ["wavelength [$\AA$]", "flux " + r"[J/(s m$^2$ m)]"])
	'''
	
# 1: POWERLAW
elif sed == 1:
	wave_ricevuta = (1 + z) * wave_sorgente
	flusso_energia = sd.get_powerlaw(wave_sorgente, sed_index)  # J/(s * m2 * m)
	sed_name = "Powerlaw"
	'''
	if DEBUG:
		plots.generic(wave_sorgente, [[flusso_energia, sed_name]], sed_name +
                          " SED", ["wavelength [$\AA$]", "flux " + r"[J/(s m$^2$ m)]"])
	'''


# sez 3 : photons flux and normalization & conversion in erg/sec/cm2/Ang
# sez 3.2 photons flux and normalization
nfotoni = flusso_energia * wave_ricevuta * 10 ** (-10) * 10 ** (-7) / constants.h / constants.c  # [phot/s/cm2/A]
'''
if DEBUG:
	plots.generic(wave_ricevuta, [[nfotoni, sed_name]], "TEMP", [
                      "wavelength [$\AA$]", "flux " + r"[ph/s/cm$^2$/$\AA$]"], displey_orders=True, x_cut=x_cut)
'''
# sez 3.2 normalization
fot_norm_1 = zp_norm.get_filter_norm(wave_ricevuta, nfotoni, bandm_list[bandm], zeropoint, mag)  # [phot/s/cm2/A]

# sez 3.3 conversion in erg/sec/cm2/Ang
fot_norm_2 = fot_norm_1 * ((constants.h * constants.c)/(wave_ricevuta*(10**-10))) * (10**7)
fot_norm_3 = [wave_ricevuta, fot_norm_2]
#
plt.figure()
plt.plot(wave_ricevuta, fot_norm_2, 'b')
plt.axis([wave_ricevuta[0], wave_ricevuta[-1], 0, fot_norm_2.max()])
plt.title('SED Test')
plt.show()


# sez 4 : Loading SED from database
SED_load = np.loadtxt(filepath_load, delimiter=' ', skiprows=1, dtype=float)
SED_load[:,0] = SED_load[:,0]*10 # from nm to Ang
#
plt.figure()
plt.plot(wave_ricevuta, fot_norm_2, 'b', label='Theor')
plt.plot(SED_load[:,0], SED_load[:,1], 'r', label='ESO-ETC') 
plt.axis([wave_ricevuta[0], wave_ricevuta[-1], 0, fot_norm_2.max()])
plt.title('SED comparison')
plt.legend()
plt.grid()
plt.show()


# Sec 5 : Interpolation and difference
# Sec 5.1 : interp
fot_norm_2_interp = np.interp(SED_load[:,0],wave_ricevuta, fot_norm_2)
#Sec 5.2 : difference
rel_diff = np.abs(fot_norm_2_interp - SED_load[:,1]) / SED_load[:,1]
rel_diff_mean = np.mean(rel_diff)
#
plt.figure()
plt.plot(SED_load[:,0], rel_diff, 'b',label = 'Rel Diff')
plt.axhline(rel_diff_mean, color = 'red', label=f'Mean = {rel_diff_mean*100:0.2}%')
plt.axis([SED_load[0,0], SED_load[-1,0], rel_diff.min(), rel_diff.max()])
plt.title('Relative difference')
plt.grid()
plt.legend()
plt.show()

