import numpy as np
import os
import logging
import spiceypy as spice
from glob import glob as filesMatchingPattern
import MoonMag.trajec_analysis as traj
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
logLevel = logging.DEBUG
printFmt = '[%(levelname)s] %(message)s'
logging.basicConfig(level=logLevel, format=printFmt)

bname = "Ganymede"
parent = "Jupiter"
fNameList = ["high", "low", "simple"]
sc = ["Galileo", "Juno"]
scSpice = ["GALILEO ORBITER", "JUNO"]
modelList = [f"{scN}_JRM33C2020" for scN in sc]
trajecDir = "Alain"
trajecOut = "outData"
moon = bname.upper()
spkMoon = f"IAU_{moon}"
datHeader = "UTC time, Bx_high, By_high, Bz_high, Bx_low, By_low, Bz_low, Bx_simple, By_simple, Bz_simple, x (R_G), y(R_G), z(R_G)\n"
nvals = [1, 1, 1]
mvals = [-1, 0, 1]

kPath = "spice"
kTLS = "naif0012.tls"
kPhiO = "clipper_dyn_v01_mod.tf"
kPCKgal = "pck00006.tpc"
genericGal = "jup365.bsp"
kSPKgal = ["s980326a.bsp", "s000131a.bsp", "s030916a.bsp"]
kPCKjuno = "pck00010.tpc"
genericJuno = "jup380s.bsp"
kSPKjuno = "juno_rec_210513_210630_210707.bsp"

# Body radius in km
Rlist = {
    "Ariel": 578.9,
    "Callisto": 2410.3,
    "Enceladus": 252.1,
    "Europa": 1560.0,
    "Ganymede": 2634.1,
    "Miranda": 235.8,
    "Titan": 2574.7,
    "Triton": 1353.4
}
R = Rlist[bname]

galFiles = np.sort(filesMatchingPattern(os.path.join(trajecDir, "ORB??_GAN*")))
junoFiles = np.sort(filesMatchingPattern(os.path.join(trajecDir, "ORB???_GAN*")))
tUTCgal,  etsGal,  xyzGal  = ({key:np.empty(0) for key in galFiles}  for _ in range(3))
tUTCjuno, etsJuno, xyzJuno = ({key:np.empty(0) for key in junoFiles} for _ in range(3))
BxGal,  ByGal,  BzGal  = ({key:{subkey:np.empty(0) for subkey in fNameList} for key in galFiles}  for _ in range(3))
BxJuno, ByJuno, BzJuno = ({key:{subkey:np.empty(0) for subkey in fNameList} for key in junoFiles} for _ in range(3))
BinmJ2000gal, BinmGal, omegaGal, BinmJ2000juno, BinmJuno, omegaJuno = ({key:np.empty(0) for key in fNameList} for _ in range(6))
BxG, ByG, BzG, BxJ, ByJ, BzJ = ({key:np.empty(0) for key in fNameList} for _ in range(6))
kNamesGal  = [kTLS, kPCKgal,  genericGal,  kPhiO] + kSPKgal
kNamesJuno = [kTLS, kPCKjuno, genericJuno, kPhiO, kSPKjuno]

for fType in fNameList:
    fName_BinmGal  = os.path.join("induced", f"{bname}_Binm_sym_{fType}_{modelList[0]}.dat")
    fName_BinmJuno = os.path.join("induced", f"{bname}_Binm_sym_{fType}_{modelList[1]}.dat")

    Tpeaks_h,  _, _, lin_Binm_Re, lin_Binm_Im = np.loadtxt(fName_BinmGal, skiprows=1, unpack=True, delimiter=',')
    omegaGal[fType] = 2*np.pi/3600/np.unique(Tpeaks_h)
    BinmJ2000gal[fType]  = np.reshape(lin_Binm_Re + 1j*lin_Binm_Im, (np.size(omegaGal[fType]), -1))
    Tpeaks_h, _, _, lin_Binm_Re, lin_Binm_Im = np.loadtxt(fName_BinmJuno, skiprows=1, unpack=True, delimiter=',')
    omegaJuno[fType] = 2*np.pi/3600/np.unique(Tpeaks_h)
    BinmJ2000juno[fType] = np.reshape(lin_Binm_Re + 1j*lin_Binm_Im, (np.size(omegaJuno[fType]), -1))

npts = 0
nEnds = np.empty(0)
kernels = [os.path.join(kPath, kName) for kName in kNamesGal]
spice.furnsh(kernels)
for fb in galFiles:
    log.debug(f"Evaluating times from {fb}")
    tUTCgal[fb], _, _, _, _, _, _, _ = np.loadtxt(fb, unpack=True, dtype="U23,f,f,f,f,f,f,f", delimiter="\t")
    etsGal[fb] = spice.str2et(tUTCgal[fb])
    xyzGal[fb], _ = spice.spkpos(scSpice[0], etsGal[fb], spkMoon, 'NONE', moon)
    xyzGal[fb] = xyzGal[fb] / R
    x = xyzGal[fb][:,0]
    y = xyzGal[fb][:,1]
    z = xyzGal[fb][:,2]
    r = np.sqrt(x**2 + y**2 + z**2)
    for fType in fNameList:
        BxGal[fb][fType], ByGal[fb][fType], BzGal[fb][fType] = traj.calc_trajec(x, y, z, r, etsGal[fb],
                                                               BinmJ2000gal[fType], 0, [0,0,0], omegaGal[fType],
                                                               nvals, mvals, fieldType="ind")
        BxG[fType] = np.concatenate((BxG[fType], BxGal[fb][fType]))
        ByG[fType] = np.concatenate((ByG[fType], ByGal[fb][fType]))
        BzG[fType] = np.concatenate((BzG[fType], BzGal[fb][fType]))

    with open(os.path.join(trajecOut, fb), 'w') as f:
        f.write(datHeader)
        for i,t in enumerate(tUTCgal[fb]):
            Bvals = "".join([f", {BxGal[fb][fType][i]:12.3f}, {ByGal[fb][fType][i]:12.3f}, {BzGal[fb][fType][i]:12.3f}" for fType in fNameList])
            f.write(f"{t}{Bvals}, {x[i]:12.5f}, {y[i]:12.5f}, {z[i]:12.5f}\n")
    npts += len(x)
    nEnds = np.append(nEnds, npts)
spice.kclear()

nptsGal = npts + 0
kernels = [os.path.join(kPath, kName) for kName in kNamesJuno]
spice.furnsh(kernels)
for fb in junoFiles:
    log.debug(f"Evaluating times from {fb}")
    tUTCjuno[fb], _, _, _, _, _, _, _ = np.loadtxt(fb, unpack=True, dtype="U23,f,f,f,f,f,f,f", delimiter="\t")
    etsJuno[fb] = spice.str2et(tUTCjuno[fb])
    xyzJuno[fb], _ = spice.spkpos(scSpice[1], etsJuno[fb], spkMoon, 'NONE', moon)
    xyzJuno[fb] = xyzJuno[fb] / R
    x = xyzJuno[fb][:,0]
    y = xyzJuno[fb][:,1]
    z = xyzJuno[fb][:,2]
    r = np.sqrt(x**2 + y**2 + z**2)
    for fType in fNameList:
        BxJuno[fb][fType], ByJuno[fb][fType], BzJuno[fb][fType] = traj.calc_trajec(x, y, z, r, etsJuno[fb],
                                                               BinmJ2000gal[fType], 0, [0,0,0], omegaJuno[fType],
                                                               nvals, mvals, fieldType="ind")
        BxJ[fType] = np.concatenate((BxJ[fType], BxJuno[fb][fType]))
        ByJ[fType] = np.concatenate((ByJ[fType], ByJuno[fb][fType]))
        BzJ[fType] = np.concatenate((BzJ[fType], BzJuno[fb][fType]))

    with open(os.path.join(trajecOut, fb),'w') as f:
        f.write(datHeader)
        for i,t in enumerate(tUTCjuno[fb]):
            Bvals = "".join([f", {BxJuno[fb][fType][i]:12.3f}, {ByJuno[fb][fType][i]:12.3f}, {BzJuno[fb][fType][i]:12.3f}" for fType in fNameList])
            f.write(f"{t}{Bvals}, {x[i]:12.5f}, {y[i]:12.5f}, {z[i]:12.5f}\n")
spice.kclear()


fig, axes = plt.subplots(3, 1, figsize=(6, 6))
fig.suptitle(f"Ganymede induced field components, all flybys")
axes[2].set_xlabel("Measurement index")
axes[0].set_ylabel("IAU $B_x$ (nT)")
axes[1].set_ylabel("IAU $B_y$ (nT)")
axes[2].set_ylabel("IAU $B_z$ (nT)")
c = ["blue", "green", "brown", "black", "red", "#b000ff", 'gray']

nptsGal = np.size(BxG[fNameList[0]])
npts = nptsGal + np.size(BxJ[fNameList[0]])
vThick = 0.5
jThick = 0.75
lThick = 1.0
ind = np.arange(0, npts)
for i,fType in enumerate(fNameList):
    axes[0].plot(ind, np.concatenate((BxG[fType], BxJ[fType])), color=c[i%4], label=fType, linewidth=lThick)
    axes[1].plot(ind, np.concatenate((ByG[fType], ByJ[fType])), color=c[i%4], label=fType, linewidth=lThick)
    axes[2].plot(ind, np.concatenate((BzG[fType], BzJ[fType])), color=c[i%4], label=fType, linewidth=lThick)
    axes[0].axvline(nptsGal, color=c[5], zorder=-1, linewidth=jThick)
    axes[1].axvline(nptsGal, color=c[5], zorder=-1, linewidth=jThick)
    axes[2].axvline(nptsGal, color=c[5], zorder=-1, linewidth=jThick)
    for n in nEnds[:-1]:
        axes[0].axvline(n, color=c[6], zorder=-1, linewidth=vThick)
        axes[1].axvline(n, color=c[6], zorder=-1, linewidth=vThick)
        axes[2].axvline(n, color=c[6], zorder=-1, linewidth=vThick)


axes[0].set_xlim([0,npts])
axes[1].set_xlim([0,npts])
axes[2].set_xlim([0,npts])
axes[0].legend()

xtn = "png"
thefig = os.path.join("figures", f"BindGanymedeFlybys.{xtn}")
fig.savefig(thefig, format=xtn, dpi=300)
plt.close()
log.info(f"Induced field plot printed to: {thefig}")
