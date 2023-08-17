import os
import numpy as np
import healpy as hp
import healpy.sphtfunc as hps
from healpy.projector import CartesianProj
import logging
from scipy.interpolate import RectSphereBivariateSpline
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from PlanetProfile.Plotting.MagPlots import SetMap
from PlanetProfile.GetConfig import Color, Style, FigLbl, FigSize, FigMisc
from MoonMag import _interior
import MoonMag.asymmetry_funcs as asym
from MoonMag.plotting_funcs import get_latlon

# Set up log messages
printFmt = '[%(levelname)s] %(message)s'
logHP = logging.getLogger('healpy')
stream = logging.StreamHandler()
stream.setFormatter(logging.Formatter(printFmt))
logHP.setLevel(logging.WARNING)
log = logging.getLogger('MoonMag')
log.setLevel(logging.DEBUG)

# Map plotting function
def PlotMap(plotData, title, fName, levels=None, cmap=None):
    fig = plt.figure(figsize=FigSize.asym)
    grid = GridSpec(1, 1)
    ax = fig.add_subplot(grid[0, 0])
    SetMap(ax)
    asymMap = ax.pcolormesh(lonMap_deg, latMap_deg, plotData,
                            shading='auto', cmap=cmap, rasterized=FigMisc.PT_RASTER)
    asymContours = ax.contour(lonMap_deg, latMap_deg, plotData,
                              levels=levels, colors='black')
    ax.clabel(asymContours, fmt=ticker.FuncFormatter(FigMisc.Cformat),
              fontsize=FigMisc.cLabelSize, inline_spacing=FigMisc.cLabelPad)
    ax.set_title(title, size=FigMisc.mapTitleSize)
    ax.set_aspect(1)
    cax = ax.inset_axes([1 + 0.02, 0, 0.03, 1])
    fig.colorbar(asymMap, ax=ax, cax=cax, label=FigLbl.asymCbarLabel)
    plt.tight_layout()
    fig.savefig(fName, bbox_inches='tight', format='pdf', dpi=FigMisc.dpi, metadata=FigLbl.meta)
    log.info(f'Map figure saved to file: {fName}')
    plt.close()
    return

# Title string constructor function
def MakeTitle(titleSpec, descrip, pmax):
    return f'{titleSpec} --- {descrip}, $p_\\mathrm{{max}} = {pmax}$'

# Limit pmax to recommended value
pmax = 8
# Load spherical harmonics to work with
asym_model = os.path.join(_interior, 'depth_chi_pq_shape_Enceladus.txt')
descrip = 'Enceladus ice shell'
_, _, flinCpq, flinSpq, _, _ = np.loadtxt(asym_model, skiprows=1, unpack=True, delimiter=',')
# Load into format where coefficients can be referenced with CS[p,q]
fCpq, fSpq = (np.zeros((pmax+1,pmax+1)) for _ in range(2))
for p in range(pmax+1):
    this_min = int(p*(p+1)/2)
    this_max = this_min + p+1
    fCpq[p, :p+1] = flinCpq[this_min:this_max]
    fSpq[p, :p+1] = flinSpq[this_min:this_max]

# Generate HEALpix and cylindrical mapping parameters
nside = 2**8
lonMap_deg, latMap_deg, lon_min, lon_max, _, _, nLonMap, nLatMap, _, _, _, _, _, _ = get_latlon(False)
tht, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))

# Get complex, orthonormal spherical harmonic coefficients
chipq = asym.get_chipq_from_CSpq(pmax, fCpq, fSpq)
pvalsHP, qvalsHP = hp.sphtfunc.Alm.getlm(pmax)
# Get the format HEALpix expects (linear, with coefficients for negative m discarded)
apq = asym.get_apq_from_chipq(pvalsHP, qvalsHP, chipq)
# Evaluate HEALpix map based on these coefficients
map = hps.alm2map(apq, nside)

# Convert to cylindrical projection for mapping
cp = CartesianProj(flipconv='geo')
cp.set_proj_plane_info(xsize=nLonMap, ysize=nLatMap,
    lonra=np.array([lon_min, lon_max]), latra=None)
vecFunc = lambda vec: hp.pixelfunc.vec2pix(nside, vec[0], vec[1], vec[2])
pix = vecFunc(cp.xy2vec(cp.ij2xy()))
mpix = map[pix]

# Interpolate from cylindrical projection back to HEALpix
# First, sort map values in increasing theta, as needed for RectSphereBivariateSpline
impixSort = np.argsort(tht[pix], axis=0)
mpixRect = np.take_along_axis(mpix, impixSort, axis=0)
# RectSphereBivariateSpline does not allow grid points exactly at the poles or at 2pi past the first phi value
abit = 1e-6
thtRect = np.linspace(abit, np.pi-abit, nLatMap)
phiRect = np.linspace(-np.pi, np.pi-abit, nLonMap)
# Interpolate over sphere to go from cylindrical projection back to HEALpix
remapFunc = RectSphereBivariateSpline(thtRect, phiRect, mpixRect)
phiHP = phi + 0
phiHP[phiHP > np.pi] -= 2*np.pi
remap = remapFunc(tht, phiHP, grid=False)

# Recover spherical harmonic coefficients from cylindrical projection map
apq = hp.map2alm(remap, lmax=pmax, use_pixel_weights=True)
chipq = asym.get_chipq_from_apq(pvalsHP, qvalsHP, apq)

# Manually evaluate spherical harmonics into map using MoonMag infrastructure
pvals = [p for p in range(1, pmax+1) for _ in range(-p, p+1)]
qvals = [q for p in range(1, pmax+1) for q in range(-p, p+1)]
hpAsymDevs = asym.get_rsurf(pvals, qvals, chipq, None, tht, phi)
asymDevs = hpAsymDevs[pix]

# Calculate difference map between recovered and initial spherical harmonic coefficients
hpDiff = map - hpAsymDevs
# Find the spherical harmonics represented in the difference
almDiff = hp.map2alm(hpDiff, lmax=pmax)

# Plot maps
levels = round(fCpq[0, 0]) + np.array([-15, -11, -7, -3, 1, 5, 9])
PlotMap(mpix, MakeTitle('HEALpix evaluation', descrip, pmax), 'healpix.pdf', levels=levels, cmap=Color.cmap['asymDev'])
PlotMap(asymDevs, MakeTitle('Direct cylindrical evaluation', descrip, pmax), 'direct.pdf', levels=levels, cmap=Color.cmap['asymDev'])
PlotMap(hpDiff[pix], MakeTitle('HEALpix $-$ direct', descrip, pmax), 'healpixDiff.pdf', levels=None, cmap=Color.cmap['BmapDiv'])
