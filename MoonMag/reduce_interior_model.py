import numpy as np
import os

fNameHigh = "GanymedeProfile_MgSO410WtPct_Ts110Zb25240mQm1mWm2_CMR2p3115_CM2hy1wt_678_1.txt"
fNameLow = "GanymedeProfile_MgSO41WtPct_Ts110Zb94295mQm1mWm2_CMR2p3115_CM2hy1wt_678_1.txt"
fNameSaur = "GanymedeSaur2015.txt"
fNameList = [fNameHigh, fNameLow, fNameSaur]
fNameOut = ["high", "low", "simple"]
header = "Radius (m),Conductivity (S/m),Bdy deviation (epsilon/R)\n"

for i,fName in enumerate(fNameList):
    infPath = os.path.join("interior", fName)
    outfPath = os.path.join("interior", f"interior_model_asym_Ganymede_{fNameOut[i]}.txt")
    P, T, r_km, rho, VP, VS, QS, KS, GS, g, phase, sigma = np.loadtxt(infPath, skiprows=1, unpack=True)

    iSig_nonzero = np.squeeze(np.argwhere(sigma != 0))
    if np.size(iSig_nonzero) == 1:
        r_m = np.array([r_km[0], r_km[iSig_nonzero], r_km[iSig_nonzero + 1]]) * 1e3
        sigma_Sm = np.array([1e-16, sigma[iSig_nonzero], 1e-16])
    else:
        iSig_nonzero = np.squeeze(iSig_nonzero)
        r_m = np.concatenate(([r_km[0]], r_km[iSig_nonzero], [r_km[iSig_nonzero[-1]+1]]))*1e3
        sigma_Sm = np.concatenate(([1e-16], sigma[iSig_nonzero], [1e-16]))
    npts = np.size(r_m)
    
    r_m = np.flip(r_m)
    sigma_Sm = np.flip(sigma_Sm)
    
    with open(outfPath,'w') as f:
        f.write(header)
        for j in range(npts):
            f.write(f"{r_m[j]},{sigma_Sm[j]},0\n")
