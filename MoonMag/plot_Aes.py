import numpy as np
import os
import MoonMag.asymmetry_funcs as asym
import MoonMag.plotting_funcs as plots

bname = "Ganymede"
fNameList = ["high", "low", "simple"]
modelList = ["Galileo_JRM33C2020", "Juno_JRM33C2020"]
peak_periods, Benm, BeRatioSorted, peaksSorted = ({key:np.empty(0) for key in modelList} for _ in range(4))
AeMs, AeAs = ({key:np.empty(0) for key in fNameList} for _ in range(2))

for fType in fNameList:
    fName = os.path.join("induced", f"complexAes_{fType}.dat")
    T_h, AeMs[fType], AeAs[fType] = np.loadtxt(fName, skiprows=1, delimiter=",", unpack=True)

for model in modelList:
    peak_periods[model], Benm[model], _ = asym.read_Benm(1, 0, bodyname=bname, model=model)
    BeMag = np.zeros_like(peak_periods[model])
    for i_om in range(np.size(BeMag)):
        BeMag[i_om] = np.sqrt(np.sum(abs(Benm[model][i_om,...])**2))
    BeRatio = (BeMag / np.max(BeMag))**(1/2)
    pairs = np.array(sorted(zip(BeRatio, peak_periods[model]), reverse=True))
    BeRatioSorted[model] = pairs[:,0]
    peaksSorted[model] = pairs[:,1]

    plots.plotAes(T_h, AeMs, AeAs, peaksSorted[model], BeRatioSorted[model], fNameList, fEnd=f"_{model}")



