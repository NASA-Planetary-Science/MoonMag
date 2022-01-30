""" This program calculates magnetic fields along a trajectory
    and is intended for supporting analysis of spacecraft data.
Author: M. J. Styczinski, mjstyczi@uw.edu """

import sys
import numpy as np
import field_xyz as field
import asymmetry_funcs as asym
import plotting_funcs as plots
import eval_induced_field as eval
from glob import glob as filesMatchingPattern
import spiceypy as spice
import multiprocessing as mtp
num_cores = mtp.cpu_count()
J2000 = np.datetime64("2000-01-01T11:58:55.816")

def fitData(bname, recalcMoments=True, recalcData=True, recalcFlybys=True, do_parallel=True):
    print(f" - {bname} - ")
    kPath = "spice/"
    datPath = f"{kPath}{bname}/"
    outPath = "outData/"

    kTLS = "naif0012.tls"
    kPCK = "pck00010.tpc"
    kNames = [kTLS, kPCK]

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

    # Amplitude of white noise to add to EACH vector component, in nT
    noiseAmp_nT = 0.5

    # Calculate the induced magnetic moments for this interior model
    if recalcMoments:
        eval.run_calcs(bname, None, True, True, False)

    if bname == "Europa":
        # Get list of usable Galileo flyby reference names
        fbList = np.array(["e4", "e11", "e12", "e14", "e15", "e19", "e26"])
        # Get list of Galileo flyby closest approach (CA) times
        CAlist = ["1996-12-19T06:52:58.000",
                  "1997-11-06T20:31:44.000",
                  "1997-12-16T12:03:20.000",
                  "1998-03-29T13:21:06.000",
                  "1998-05-31T21:12:58.000",
                  "1999-02-01T02:19:50.000",
                  "2000-01-03T17:59:44.000"]
        kNames.append("jup365.bsp")
        kNames.append("s980326a.bsp")
        kNames.append("s000131a.bsp")
        parent = "JUPITER"
        sc = "GALILEO ORBITER"
    else:
        raise ValueError("Only Europa is currently implemented for flyby data analysis.")
    CAlist = np.array([np.datetime64(CAi) for CAi in CAlist])
    nFlybys = np.size(fbList)

    t, tRel, x, y, z, r, BxDat, ByDat, BzDat, BxFit, ByFit, BzFit = (np.empty((nFlybys,), dtype=object) for _ in range(12))

    tDescrip = "Measurement datetime"
    if recalcData:
        print("Loading flyby data with SPICE")

        # Load spice kernels and prep for querying them
        moon = bname.upper()
        spkParent = f"IAU_{parent}"
        spkMoon = f"IAU_{moon}"
        kFiles = [f"{kPath}{kName}" for kName in kNames]
        spice.furnsh(kFiles)

        for i, thisFlyby in np.ndenumerate(fbList):
            datName = f"{datPath}{thisFlyby}-mag-sys3.tab"
            t[i], Br, Bth, Bphi, _, _, _, _, _ = \
                np.loadtxt(datName, unpack=True, dtype=("U23,f,f,f,f,f,f,f,f"))
            print(f"Loaded flyby data for {datName}")
            tRel[i] = spice.str2et(t[i])
            t[i] = np.array([np.datetime64(ti) for ti in t[i]])
            pos, _ = spice.spkpos(sc, tRel[i], spkMoon, 'NONE', moon)
            x[i] = pos[:,0] / R
            y[i] = pos[:,1] / R
            z[i] = pos[:,2] / R
            r[i] = np.sqrt(x[i]**2 + y[i]**2 + z[i]**2)
            posS3, _ = spice.spkpos(sc, tRel[i], spkParent, 'NONE', parent)
            rS3 = np.sqrt(posS3[:,0]**2 + posS3[:,1]**2 + posS3[:,2]**2)
            th = np.arccos(posS3[:,2] / rS3)
            phi = np.arctan2(posS3[:,1], posS3[:,0])
            BxyzS3 = Bsph2Bxyz(Br, Bth, Bphi, th, phi)
            npts = np.size(tRel[i])
            # Get rotation matrix with pxform to convert Sys3 xyz to moon-centered IAU xyz,
            # then multiply by each B vector to save on needing to keep the rotation matrix.
            # pxform does not accept an array of ets in spiceypy.
            Bxyz = np.array(
                [spice.mxv(spice.pxform(spkParent, spkMoon, tRel[i][ii]), BxyzS3[ii,:]) for ii in range(npts)])
            BxDat[i] = Bxyz[:,0]
            ByDat[i] = Bxyz[:,1]
            BzDat[i] = Bxyz[:,2]

            outDatName = f"{datPath}{thisFlyby}-mag-IAU.tab"
            saveDat(t[i], x[i], y[i], z[i], BxDat[i], ByDat[i], BzDat[i], tDescrip, datName=outDatName)
    else:
        print("Reloading flyby data from disk")
        for i, thisFlyby in np.ndenumerate(fbList):
            datName = f"{datPath}{thisFlyby}-mag-IAU.tab"
            t[i], x[i], y[i], z[i], BxDat[i], ByDat[i], BzDat[i] = np.loadtxt(datName, unpack=True, skiprows=1,
                                                                              delimiter=',', dtype=("U23,f,f,f,f,f,f"))
            r[i] = np.sqrt(x[i]**2 + y[i]**2 + z[i]**2)
            spice.furnsh(kPath + kTLS)
            tRel[i] = spice.str2et(t[i])
            t[i] = np.array([np.datetime64(ti) for ti in t[i]])
            print(f"Loaded flyby data {datName}")

    # We need to recalc best fit solutions if we recalc data transformations
    if recalcFlybys or recalcData:
        print("Calculating fields along trajectories")
        nprm_max = 1
        p_max = 2
        peak_periods, Benm, B0 = asym.read_Benm(nprm_max, p_max, bodyname=bname, synodic=False, orbital=False)
        peak_omegas = 2 * np.pi / (peak_periods * 3600)

        BinmFiles = filesMatchingPattern(f"induced/{bname}_Binm_asym_Tobie*")
        nBinms = np.size(BinmFiles)
        if nBinms <= 10:
            print(f"Running trajectories for induced moments calculations:")
            [print(BinmFile) for BinmFile in BinmFiles]

        if do_parallel:
            if nBinms < num_cores:
                tooFewProc = True
                print("The number of induced field models to evaluate is fewer than the " +
                      "number of processors available, but do_parallel is set to True. " +
                      "Best fit calculations will be done serially to avoid process spawning " +
                      "slowdowns.")
            else:
                tooFewProc = False
        else:
            tooFewProc = False

        Bx, By, Bz = (np.empty((nFlybys, nBinms), dtype=object) for _ in range(3))
        dS = np.zeros((nFlybys,nBinms))
        Binm, nvals, mvals = (np.empty((nBinms,), dtype=object) for _ in range(3))
        for j in range(nBinms):
            _, nvals[j], mvals[j], lin_Binm_Re, lin_Binm_Im = np.loadtxt(BinmFiles[j], skiprows=1, unpack=True,
                                                                         delimiter=',')
            Binm[j] = np.reshape(lin_Binm_Re + 1j * lin_Binm_Im, (np.size(peak_omegas), -1))
            nvals[j] = nvals[j].astype(np.int_)
            mvals[j] = mvals[j].astype(np.int_)

        for i, thisFlyby in np.ndenumerate(fbList):
            print(f"Getting square differences for {thisFlyby.upper()}, #{i[0]+1} of {nFlybys}.")
            if do_parallel and not tooFewProc:
                pool = mtp.Pool(num_cores)
                par_result = [pool.apply_async(fit_trajec, (x[i], y[i], z[i], r[i], tRel[i],
                                                            BxDat[i], ByDat[i], BzDat[i], Binm[j], Benm, B0,
                                                            peak_omegas, nvals[j], mvals[j]))
                              for j in range(nBinms)]
                pool.close()
                pool.join()

                for j in range(nBinms):
                    Bx[i,j], By[i,j], Bz[i,j], dS[i,j] = par_result[j].get()
            else:
                for j in range(nBinms):
                    Bx[i,j], By[i,j], Bz[i,j], dS[i,j] = fit_trajec(x[i], y[i], z[i], r[i], tRel[i],
                                                                    BxDat[i], ByDat[i], BzDat[i], Binm[j], Benm, B0,
                                                                    peak_omegas, nvals[j], mvals[j])

        S = np.array([np.sum(dS[:,j]) for j in range(nBinms)])
        chi2 = S / np.sum([np.size(r[i]) for i in range(nFlybys)])
        jFit = np.argmin(chi2)
        print(f"Plotting fits for best-fit chi-squared value of {chi2[jFit]:.1f}.")
        for i, thisFlyby in np.ndenumerate(fbList):
            BxFit[i] = Bx[i,jFit][0]
            ByFit[i] = By[i,jFit][0]
            BzFit[i] = Bz[i,jFit][0]
            datName = f"{outPath}{bname}-{thisFlyby.upper()}-MAG-IAU-bestFit.dat"
            saveDat(t[i], x[i], y[i], z[i], BxFit[i], ByFit[i], BzFit[i], tDescrip, datName=datName)
    else:
        print("Reloading best fit data from disk")
        for i, thisFlyby in np.ndenumerate(fbList):
            fitName = f"{outPath}{bname}-{thisFlyby.upper()}-MAG-IAU-bestFit.dat"
            _, x[i], y[i], z[i], BxFit[i], ByFit[i], BzFit[i] = np.loadtxt(fitName, unpack=True, skiprows=1,
                                                                              delimiter=',', dtype=("U23,f,f,f,f,f,f"))
            print(f"Loaded fit data {fitName}")

    for i, thisFlyby in np.ndenumerate(fbList):
        plots.plotTrajec(t[i], BxFit[i], ByFit[i], BzFit[i], Bdat=(BxDat[i], ByDat[i], BzDat[i]),
                         bodyname=bname, t_CA=CAlist[i], append=f"{thisFlyby.upper()}-bestFit",
                         fpath=outPath)


def calc_trajec(x,y,z,r,t, Binm, Benm, B0, peak_omegas, nvals, mvals, nprm_max=1, n_max=1,
                fieldType="net", noiseAmp=None):
    if n_max > 4:
        n_max = 4
        print("WARNING: Evaluation of magnetic fields is supported only up to n=4. n_max has been set to 4.")

    if fieldType == "ind":
        Benm = 0
    elif fieldType == "ext":
        Binm = 0

    Nnm = (n_max + 1) ** 2 - 1
    Nnmprm = (nprm_max + 1) ** 2 - 1
    n_peaks = np.size(peak_omegas)
    n_pts = np.size(t)

    # Linearize Binm values
    if np.size(np.shape(Binm)) > 2:
        lin_Binm = np.zeros((n_peaks,Nnm), dtype=np.complex_)
        for i_om in range(n_peaks):
            lin_Binm[i_om,:] = np.array([ Binm[i_om,int(mvals[iN]<0),nvals[iN],abs(mvals[iN])] for iN in range(Nnm) ])
    else:
        lin_Binm = Binm

    Bnet_x, Bnet_y, Bnet_z = (np.zeros(n_pts, dtype=np.complex_) for _ in range(3))

    for i_om in range(n_peaks):
        for iN in range(Nnm):
            n = nvals[iN]
            m = mvals[iN]
            if iN < Nnmprm and fieldType != "ind":
                Be_x, Be_y, Be_z = field.eval_Be(n, m, Benm[i_om,int(m<0),n,abs(m)], x, y, z, r, omega=peak_omegas[i_om], t=t)
                Bnet_x += Be_x
                Bnet_y += Be_y
                Bnet_z += Be_z

            Bi_x, Bi_y, Bi_z = field.eval_Bi(n, m, lin_Binm[i_om,iN], x, y, z, r, omega=peak_omegas[i_om], t=t)
            Bnet_x += Bi_x
            Bnet_y += Bi_y
            Bnet_z += Bi_z

    # Add uniform background field
    Bnet_x += B0[0]
    Bnet_y += B0[1]
    Bnet_z += B0[2]

    Bx = np.real(Bnet_x)
    By = np.real(Bnet_y)
    Bz = np.real(Bnet_z)

    if noiseAmp is not None:
        Bxnoise = np.random.normal(0, noiseAmp, np.size(Bx))
        Bynoise = np.random.normal(0, noiseAmp, np.size(By))
        Bznoise = np.random.normal(0, noiseAmp, np.size(Bz))
    else:
        Bxnoise = 0
        Bynoise = 0
        Bznoise = 0

    Bx += Bxnoise
    By += Bynoise
    Bz += Bznoise

    return Bx, By, Bz


def fit_trajec(x, y, z, r, tRel, BxDat, ByDat, BzDat, Binm, Benm, B0, peak_omegas, nvals, mvals):
    Bx, By, Bz = calc_trajec(x, y, z, r, tRel, Binm, Benm, B0, peak_omegas, nvals, mvals,
                             nprm_max=1, n_max=3, fieldType="net")

    BxDiff = Bx - BxDat
    ByDiff = By - ByDat
    BzDiff = Bz - BzDat
    dS = np.sum(BxDiff**2 + ByDiff**2 + BzDiff**2)

    return [Bx], [By], [Bz], dS


def Bsph2Bxyz(Br, Bth, Bphi, th, phi):
    npts = np.size(Br)
    Bxyz = np.zeros((npts,3))
    Bxyz[:,0] =  np.sin(th)  * np.cos(phi) * Br \
               + np.cos(th)  * np.cos(phi) * Bth \
               - np.sin(phi) * Bphi
    Bxyz[:,1] =  np.sin(th)  * np.sin(phi) * Br \
               + np.cos(th)  * np.sin(phi) * Bth \
               + np.cos(phi) * Bphi
    Bxyz[:,2] =  np.cos(th)  * Br \
               - np.sin(th)  * Bth

    return Bxyz


def saveDat(t, x, y, z, Bx, By, Bz, tDescrip, datName="trajectory"):
    header = " ".join([f"{tDescrip},".ljust(25),
                       f"X (R_{bname[0]}),".ljust(19),
                       f"Y (R_{bname[0]}),".ljust(19),
                       f"Z (R_{bname[0]}),".ljust(19),
                       "Bx (nT),".ljust(19),
                       "By (nT),".ljust(19),
                       "Bz (nT)\n"])
    with open(datName, 'w') as fdat:
        fdat.write(header)
        for i, ti in np.ndenumerate(t):
            fdat.write(f"{ti}, {x[i]:17.10e}, {y[i]:17.10e}, {z[i]:17.10e}, {Bx[i]:17.10e}, {By[i]:17.10e}, {Bz[i]:17.10e}\n")
    print(f"Trajectory data saved to file: {datName}")

    return


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Body name was passed as command line argument
        bname = sys.argv[1]
    else:
        print("No body name entered. Defaulting to Europa.")
        bname = "Europa"
    fitData(bname, recalcMoments=False, recalcData=True, recalcFlybys=True, do_parallel=False)
