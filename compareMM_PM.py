import os
import numpy as np
import logging
from scipy.io import loadmat
from MoonMag.field_xyz import eval_Bi, eval_Bi_Schmidt
from MoonMag.asymmetry_funcs import get_Binm_from_gh
from MoonMag.plotting_funcs import healpixMap


def GetxyzrFromData(data):
    # Retrieve cartesian locations in units of R_P from (r,theta,phi)
    r_RP = np.squeeze(data['r_RP'])
    theta_rad = np.squeeze(data['theta_rad'])
    phi_rad = np.squeeze(data['phi_rad'])
    x_RP = r_RP * np.sin(theta_rad) * np.cos(phi_rad)
    y_RP = r_RP * np.sin(theta_rad) * np.sin(phi_rad)
    z_RP = r_RP * np.cos(theta_rad)
    return x_RP, y_RP, z_RP, r_RP


def compareMM_PM(nside=2**6, ghVal=1.0*1e5, atol=1e-8, rtol=1e-5, PLOT_DIFFS=True, PLOT_ONLY_SIGNIF=False):
    """
    Compare multipole magnetic field calculations by MoonMag and PlanetMag for the same magnetic moments
    and at the same points.

    To compare outputs:

        #. Run print_healpix_locs.py to print a text file of HEALpix locations on a sphere
        #. Run PlanetMag/OutputHEALpixTo_n10.m using Matlab. Requires healpix_locs.txt, expected
           to be at ~/MoonMag/outData/healpix_locs.txt
        #. Run this script. Requires output data from PlanetMag, expected to be at
           ~/PlanetMag/out/pureHarmMap_n##m##x.mat, where ## are harmonic indices and x is g or h.

    Parameters
    ----------
    nside : int, default=2**6
        Resolution of surface for comparison. 12 * nside**2 is the total number of pixels.
        Must match the nside value used in print_healpics_locs.
    ghVal : float, default=1.0*1e5
        Strength of magnetic moments to use in comparison in nT.
        Must match ghVal from PlanetMag printout script (times 10^5 to convert from G to nT).
    atol : float, default=1e-8
        Absolute tolerance to use in comparisons with np.isclose()
    rtol : float, default=1e-5
        Relative tolerance to use in comparisons with np.isclose()
    PLOT_DIFFS : bool, default=True
        Whether to generate map figures showing differences
    PLOT_ONLY_SIGNIF : bool, default=False
        Whether to limit printed maps to those harmonics identified as differing
    """

    zeros = np.zeros((11, 11), dtype=np.complex_)

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
            if PLOT_DIFFS:
                if (not gMatch) or not PLOT_ONLY_SIGNIF:
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
            if PLOT_DIFFS:
                if (not gMatchOrtho) or not PLOT_ONLY_SIGNIF:
                    # Plot vector component differences
                    fName = os.path.join('figures', f'MMOdiffPM_n{n:02}m{m:02}g')
                    healpixMap(nside, BvecOrtho[0,:] - gData['Bvec'][0,:], f'$B_x$ MoonMagOrtho $-$ PlanetMag, $n,m = {n:d},{m:d}$ g', f'{fName}x.pdf', cmap='seismic')
                    healpixMap(nside, BvecOrtho[1,:] - gData['Bvec'][1,:], f'$B_y$ MoonMagOrtho $-$ PlanetMag, $n,m = {n:d},{m:d}$ g', f'{fName}y.pdf', cmap='seismic')
                    healpixMap(nside, BvecOrtho[2,:] - gData['Bvec'][2,:], f'$B_z$ MoonMagOrtho $-$ PlanetMag, $n,m = {n:d},{m:d}$ g', f'{fName}z.pdf', cmap='seismic')

            if m == 0:
                hMatch = True
                hMatchOrtho = True
            else:
                # Load in h data from PlanetMag
                hData = loadmat(f'{thisFname}h.mat')
                # Get evaluation locations
                x, y, z, r = GetxyzrFromData(hData)

                # Compare h using Gauss coefficient calculation
                BvecSchmidt = np.real(np.vstack(eval_Bi_Schmidt(n, m, 0.0, ghVal, x, y, z, r)))
                hMatch = np.all(np.isclose(BvecSchmidt, hData['Bvec'], rtol, atol))
                if PLOT_DIFFS:
                    if (not hMatch) or not PLOT_ONLY_SIGNIF:
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
                if PLOT_DIFFS:
                    if (not hMatchOrtho) or not PLOT_ONLY_SIGNIF:
                        # Plot vector component differences
                        fName = os.path.join('figures', f'MMOdiffPM_n{n:02}m{m:02}h')
                        healpixMap(nside, BvecOrtho[0,:] - hData['Bvec'][0,:], f'$B_x$ MoonMagOrtho $-$ PlanetMag, $n,m = {n:d},{m:d}$ h', f'{fName}x.pdf', cmap='seismic')
                        healpixMap(nside, BvecOrtho[1,:] - hData['Bvec'][1,:], f'$B_y$ MoonMagOrtho $-$ PlanetMag, $n,m = {n:d},{m:d}$ h', f'{fName}y.pdf', cmap='seismic')
                        healpixMap(nside, BvecOrtho[2,:] - hData['Bvec'][2,:], f'$B_z$ MoonMagOrtho $-$ PlanetMag, $n,m = {n:d},{m:d}$ h', f'{fName}z.pdf', cmap='seismic')

            # Print line to console
            log.info(f'{n:>3}{m:>3}{yesInd if gMatch else noInd:>6}{yesInd if hMatch else noInd:>6}{yesInd if gMatchOrtho else noInd:>6}{yesInd if hMatchOrtho else noInd:>6}')


if __name__ == '__main__':
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

    fh = logging.FileHandler('pureHarmComp.txt')
    fh.setLevel(logging.INFO)
    log.addHandler(fh)

    compareMM_PM()
