import os
import numpy as np
import healpy as hp
import logging
from MoonMag.plotting_funcs import get_latlon

# Set up log messages
printFmt = '[%(levelname)s] %(message)s'
logHP = logging.getLogger('healpy')
stream = logging.StreamHandler()
stream.setFormatter(logging.Formatter(printFmt))
logHP.setLevel(logging.WARNING)
log = logging.getLogger('MoonMag')
log.setLevel(logging.DEBUG)

def printHealpixLocs(fName='healpix_locs.txt', outDir='outDir', nside=2**6):
    """
    Save an output file containing evaluation points for a HEALpix scheme over a sphere.

    Prints a list of (theta, phi) pairs aligned to HEALpix pixels for a given Nside value. Intended
    to support validation between MoonMag and PlanetMag for spherical harmonic calculations of
    magnetic fields on a sphere.

    Parameters
    ----------
    fName : str, default='healpix_locs.txt'
        Output file name to which to print pixel locations.
    outDir : str, default='outDir'
        Output directory to which to print pixel locations.
    nside : int, default=2**6
        Resolution of surface to print. 12 * nside**2 is the total number of pixels.
        The nside value used in compareMM_PM must match this value.
    """

    # Set output file path
    outFile = os.path.join(outDir, fName)

    # Generate HEALpix and mapping parameters
    npix = hp.nside2npix(nside)
    lonMap_deg, latMap_deg, lon_min, lon_max, _, _, nLonMap, nLatMap, _, _, _, _, _, _ = get_latlon(False)
    theta_rad, phi_rad = hp.pix2ang(nside, np.arange(npix))

    # Header strings
    header_info = f'Contains (theta, phi) pairs for HEALpix pixel locations with Nside = {nside} ({npix} pix).\n'
    thetaDescrip = 'Colatitude (rad)'
    phiDescrip = 'E longitude (rad)'
    header_cols = f'{thetaDescrip:>24},{phiDescrip:>24}\n'

    log.info(f'Printing locations for Nside = {nside} ({npix} pix).')
    with open(outFile, 'w') as f:
        f.write(header_info)
        f.write(header_cols)
        for tht, phi in zip(theta_rad, phi_rad):
            f.write(f'{tht:24.14f},{phi:24.14f}\n')


if __name__ == '__main__':

    nside = 2**6
    outDir = 'outData'
    fName = 'healpix_locs.txt'

    printHealpixLocs(fName=fName, outDir=outDir, nside=nside)