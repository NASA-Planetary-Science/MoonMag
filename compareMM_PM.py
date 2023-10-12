# Prints a list of (theta, phi) pairs aligned to HEALpix pixels for a given Nside value.

import os
import numpy as np
import healpy as hp
import logging

# Set up log messages
printFmt = '[%(levelname)s] %(message)s'
logHP = logging.getLogger('healpy')
stream = logging.StreamHandler()
stream.setFormatter(logging.Formatter(printFmt))
logHP.setLevel(logging.WARNING)
log = logging.getLogger('MoonMag')
log.setLevel(logging.DEBUG)

# HEALpix settings
nside = 2**6  # Must match that in print_healpix_locs.py

# Header strings for output
nDescrip = 'n'
mDescrip = 'm'
gDescrip = 'gnm?'
hDescrip = 'hnm?'
yesInd = '.'
noInd = '@'
header_cols = f'{nDescrip:>5}{mDescrip:>5}{gDescrip:>5}{hDescrip:>5}\n'

log.info(f'Printing locations for Nside = {nside} ({npix} pix).')
with open(outFile, 'w') as f:
    f.write(header_info)
    f.write(header_cols)
    for tht, phi in zip(theta_rad, phi_rad):
        f.write(f'{tht:24.14f},{phi:24.14f}\n')