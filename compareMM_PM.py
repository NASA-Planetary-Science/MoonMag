# Compares MoonMag- and PlanetMag-evaluated .

import os
import numpy as np
import logging
from scipy.io import loadmat
from MoonMag.field_xyz import eval_Bi, eval_Bi_Schmidt
from MoonMag.asymmetry_funcs import get_Binm_from_gh
from MoonMag.plotting_funcs import healpixMap

# Set up log messages
printFmt = '[%(levelname)s] %(message)s'
logHP = logging.getLogger('healpy')
stream = logging.StreamHandler()
stream.setFormatter(logging.Formatter(printFmt))
logHP.setLevel(logging.DEBUG)
log = logging.getLogger('MoonMag')
log.setLevel(logging.INFO)
logPP = logging.getLogger('PlanetProfile')
logPP.setLevel(logging.INFO)

def GetxyzrFromData(data):
    # Retrieve cartesian locations in units of R_P from (r,theta,phi)
    r_RP = np.squeeze(data['r_RP'])
    theta_rad = np.squeeze(data['theta_rad'])
    phi_rad = np.squeeze(data['phi_rad'])
    x_RP = r_RP * np.sin(theta_rad) * np.cos(phi_rad)
    y_RP = r_RP * np.sin(theta_rad) * np.sin(phi_rad)
    z_RP = r_RP * np.cos(theta_rad)
    return x_RP, y_RP, z_RP, r_RP

# Comparison settings
nside = 2**6  # Must match that in print_healpix_locs.py
ghVal = 1.0 * 1e5  # Must match ghVal from PlanetMag printout script (times 10^5 to convert from G to nT)
atol = 1e-8  # Absolute tolerance to use in comparisons with np.isclose
rtol = 1e-5  # Relative tolerance to use in comparisons with np.isclose
zeros = np.zeros((11,11), dtype=np.complex_)

# Header strings for formatted output
nDescrip = 'n'
mDescrip = 'm'
gDescrip = 'gnm?'
hDescrip = 'hnm?'
gOrthoDescrip = 'gOnm?'
hOrthoDescrip = 'hOnm?'
yesInd = '.'
noInd = '@'
header_cols = f'{nDescrip:>3}{mDescrip:>3}{gDescrip:>6}{hDescrip:>6}{gOrthoDescrip:>6}{hOrthoDescrip:>6}'

# File name generation
dataFnameBase = os.path.join(os.path.expanduser('~'), 'PlanetMag', 'out', 'pureHarmMap')

log.info(f'Comparing maps.\n')
log.info(header_cols)
for n in range(1, 11):
    for m in range(n+1):
        thisFname = f'{dataFnameBase}_n{n:02}m{m:02}'

        # Load in g data from PlanetMag
        gData = loadmat(f'{thisFname}g.mat')
        # Get evaluation locations
        x, y, z, r = GetxyzrFromData(gData)

        # Compare g using Gauss coefficient calculation
        BvecSchmidt = np.real(np.vstack(eval_Bi_Schmidt(n, m, ghVal, 0.0, x, y, z, r)))
        gMatch = np.all(np.isclose(BvecSchmidt, gData['Bvec'], rtol, atol))
        # Plot vector component differences
        fName = os.path.join('figures', f'MMSdiffPM_n{n:02}m{m:02}g')
        healpixMap(nside, BvecSchmidt[0,:] - gData['Bvec'][0,:], f'$B_x$ MoonMagSch $-$ PlanetMag, $n,m = {n:d},{m:d}$ g', f'{fName}x.pdf', cmap='seismic')
        healpixMap(nside, BvecSchmidt[1,:] - gData['Bvec'][1,:], f'$B_y$ MoonMagSch $-$ PlanetMag, $n,m = {n:d},{m:d}$ g', f'{fName}y.pdf', cmap='seismic')
        healpixMap(nside, BvecSchmidt[2,:] - gData['Bvec'][2,:], f'$B_z$ MoonMagSch $-$ PlanetMag, $n,m = {n:d},{m:d}$ g', f'{fName}z.pdf', cmap='seismic')

        # Prep for orthonormal calculation
        gBlock = zeros + 0.0
        gBlock[n,m] = ghVal
        gBinm = get_Binm_from_gh(10, gBlock, zeros)
        # Compare g using orthonormal calculation
        BvecOrtho = np.real(np.vstack(eval_Bi(n, m, gBinm[0,n,m], x, y, z, r)) + np.vstack(eval_Bi(n, -m, gBinm[1,n,m], x, y, z, r)))
        gMatchOrtho = np.all(np.isclose(BvecOrtho, gData['Bvec'], rtol, atol))
        # Plot vector component differences
        fName = os.path.join('figures', f'MMOdiffPM_n{n:02}m{m:02}g')
        healpixMap(nside, BvecOrtho[0,:] - gData['Bvec'][0,:], f'$B_x$ MoonMagOrtho $-$ PlanetMag, $n,m = {n:d},{m:d}$ g', f'{fName}x.pdf', cmap='seismic')
        healpixMap(nside, BvecOrtho[1,:] - gData['Bvec'][1,:], f'$B_y$ MoonMagOrtho $-$ PlanetMag, $n,m = {n:d},{m:d}$ g', f'{fName}y.pdf', cmap='seismic')
        healpixMap(nside, BvecOrtho[2,:] - gData['Bvec'][2,:], f'$B_z$ MoonMagOrtho $-$ PlanetMag, $n,m = {n:d},{m:d}$ g', f'{fName}z.pdf', cmap='seismic')

        # Load in h data from PlanetMag
        hData = loadmat(f'{thisFname}h.mat')
        # Get evaluation locations
        x, y, z, r = GetxyzrFromData(hData)

        # Compare h using Gauss coefficient calculation
        BvecSchmidt = np.real(np.vstack(eval_Bi_Schmidt(n, m, 0.0, ghVal, x, y, z, r)))
        hMatch = np.all(np.isclose(BvecSchmidt, hData['Bvec'], rtol, atol))
        # Plot vector component differences
        fName = os.path.join('figures', f'MMSdiffPM_n{n:02}m{m:02}h')
        healpixMap(nside, BvecSchmidt[0,:] - hData['Bvec'][0,:], f'$B_x$ MoonMagSch $-$ PlanetMag, $n,m = {n:d},{m:d}$ h', f'{fName}x.pdf', cmap='seismic')
        healpixMap(nside, BvecSchmidt[1,:] - hData['Bvec'][1,:], f'$B_y$ MoonMagSch $-$ PlanetMag, $n,m = {n:d},{m:d}$ h', f'{fName}y.pdf', cmap='seismic')
        healpixMap(nside, BvecSchmidt[2,:] - hData['Bvec'][2,:], f'$B_z$ MoonMagSch $-$ PlanetMag, $n,m = {n:d},{m:d}$ h', f'{fName}z.pdf', cmap='seismic')

        # Prep for orthonormal calculation
        hBlock = zeros + 0.0
        hBlock[n,m] = ghVal
        hBinm = get_Binm_from_gh(10, zeros, hBlock)
        # Compare h using orthonormal calculation
        BvecOrtho = np.real(np.vstack(eval_Bi(n, m, hBinm[0,n,m], x, y, z, r)) + np.vstack(eval_Bi(n, -m, hBinm[1,n,m], x, y, z, r)))
        hMatchOrtho = np.all(np.isclose(BvecOrtho, hData['Bvec'], rtol, atol))
        # Plot vector component differences
        fName = os.path.join('figures', f'MMOdiffPM_n{n:02}m{m:02}h')
        healpixMap(nside, BvecOrtho[0,:] - hData['Bvec'][0,:], f'$B_x$ MoonMagOrtho $-$ PlanetMag, $n,m = {n:d},{m:d}$ h', f'{fName}x.pdf', cmap='seismic')
        healpixMap(nside, BvecOrtho[1,:] - hData['Bvec'][1,:], f'$B_y$ MoonMagOrtho $-$ PlanetMag, $n,m = {n:d},{m:d}$ h', f'{fName}y.pdf', cmap='seismic')
        healpixMap(nside, BvecOrtho[2,:] - hData['Bvec'][2,:], f'$B_z$ MoonMagOrtho $-$ PlanetMag, $n,m = {n:d},{m:d}$ h', f'{fName}z.pdf', cmap='seismic')

        # Print line to console
        log.info(f'{n:>3}{m:>3}{yesInd if gMatch else noInd:>5}{yesInd if hMatch else noInd:>5}{yesInd if gMatchOrtho else noInd:>5}{yesInd if hMatchOrtho else noInd:>5}')