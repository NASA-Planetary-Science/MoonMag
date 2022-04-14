import numpy as np
import os
import MoonMag.symmetry_funcs as sym
import MoonMag.asymmetry_funcs as asym
import MoonMag.plotting_funcs as plots

output_Schmidt = 1

bname = "Ganymede"
fNameList = ["high", "low", "simple"]
modelList = ["Galileo_JRM33C2020", "Juno_JRM33C2020"]
nTypes = np.size(fNameList)
nModels = np.size(modelList)
peak_periods, peak_omegas, Benm, BeRatioSorted, peaksSorted = ({key:np.empty(0) for key in modelList} for _ in range(5))
nprm_max = 1
nprmvals = [n for n in range(1, nprm_max+1) for _ in range(-n, n+1)]
mprmvals = [m for n in range(1, nprm_max+1) for m in range(-n, n+1)]
    
periodList = np.logspace(np.log10(1e0), np.log10(1e3), 300)
omegaList = 2*np.pi/3600/periodList

for model in modelList:
    peak_periods[model], Benm[model], _ = asym.read_Benm(1, 0, bodyname=bname, model=model)
    peak_omegas[model] = 2 * np.pi / (peak_periods[model] * 3600)
    BeMag = np.zeros_like(peak_periods[model])
    for i_om in range(np.size(BeMag)):
        BeMag[i_om] = np.sqrt(np.sum(abs(Benm[model][i_om,...])**2))
    BeRatio = BeMag / np.max(BeMag)
    pairs = np.array(sorted(zip(BeRatio, peak_periods[model]), reverse=True))
    BeRatioSorted[model] = pairs[:,0]
    peaksSorted[model] = pairs[:,1]

for fType in fNameList:
    intModel = os.path.join("interior", f"interior_model_asym_Ganymede_{fType}.txt")

    r_bds, sigmas, _ = np.loadtxt(intModel, skiprows=1, unpack=True, delimiter=',')
    n_bds = np.size(r_bds)
    rscale_moments = 1/r_bds[-1]

    _, _, _ = sym.InducedAeList(r_bds, sigmas, omegaList, rscale_moments,
                                writeout=True, path="induced/", append=f"_{fType}")

    for model in modelList:
        n_peaks = np.size(peak_periods[model])
        fEnd = f"_{fType}_{model}"
        _ = sym.BiList(r_bds, sigmas, peak_omegas[model], Benm[model],
                                 nprmvals, mprmvals, rscale_moments, n_max=nprm_max,
                                 bodyname=bname, append=fEnd,
                                 output_Schmidt=output_Schmidt)
