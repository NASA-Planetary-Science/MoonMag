""" This program runs calculations for induced magnetic fields
    from near-spherical conductors and plots the results.
    Developed in Python 3.8 for "An analytic solution for evaluating the magnetic
    field induced from an arbitrary, asymmetric ocean world" by Styczinski et al.
    DOI: TBD
Author: M.J. Styczinski, mjstyczi@uw.edu """

import sys
from typing import List

import numpy as np
from collections.abc import Iterable
import matplotlib.pyplot as plt
import matplotlib as mpl

from skyfield import api as skyapi
from datetime import datetime as dtime
from datetime import timezone

from config import *
import symmetry_funcs as sym
import asymmetry_funcs as asym
import plotting_funcs as plots

def run_calcs(bname, comp, recalc, plot_field, plot_asym, do_large=False, seawater=True, compare_me=False, do_gif=False):

    absolute = (comp==None)   # Whether to plot the symmetric and asymmetric absolute magnitudes in addition to the desired component
    if comp is None:
        compstr = "_mag"
    else:
        compstr = "_" + comp

    if bname is None:
        bfname = ""
    else:
        bfname = "_"+bname
        bin_name = bname+"_"

    # IO file paths
    inp_path = "interior/"
    inp_Bi_path = "induced/"

    # p_max is highest degree in boundary shapes to use
    # bname_opt and sw_opt are strings to append to filenames to identify special cases
    sw_opt = ""
    if bname == "Enceladus":
        p_max_main = 8
        bname_opt = ""
        eval_r = 1.1 # About 25 km altitude
    elif bname == "Miranda":
        p_max_main = 8
        bname_opt = ""
        eval_r = eval_radius
    elif bname == "Europa":
        if compare_me:
            p_max_main = 2
            bname_opt = "_prev"
            eval_r = 1.0
        else:
            p_max_main = 4
            bname_opt = "_Tobie"
            if seawater:
                sw_opt = "_high"
            else:
                sw_opt = "_low"
            eval_r = 1.016 # 25 km altitude, the planned CA for some Clipper flybys
    else:
        p_max_main = 2
        bname_opt = ""
        eval_r = eval_radius

    # Override config setting in the case of previous work comparison
    if bname == "Europa" and compare_me:
        synodic_only = True
    else:
        synodic_only = synodic_period_only

    # Highest degree of induced moments
    n_max_main = nprm_max_main + p_max_main

    # Set time to evaluate magnetic fields
    tscale = skyapi.load.timescale()
    J2000 = tscale.J(2000)

    tzone = timezone.utc  # Timezone code readable by skyfield module to apply to eval_datetime string in config file -- see https://rhodesmill.org/skyfield/time.html#utc-and-your-timezone
    t = dtime.fromisoformat(eval_datetime).replace(tzinfo=tzone) # Evaluate at the specified eval_datetime in config file, in the above time zone
    #t = tscale.now().utc_datetime() # Evaluate as the bodies are right NOW
    #t = J2000.utc_datetime() # Evaluate AT J2000

    tdiff_jd = ( tscale.from_datetime(t) - J2000 )
    tsec = tdiff_jd * 86400

    # Linear arrays of values to loop over for nprm,mprm,p,q,n,m
    nprmvals = [ nprm for nprm in range(1,nprm_max_main+1) for _    in range(-nprm,nprm+1) ]
    mprmvals = [ mprm for nprm in range(1,nprm_max_main+1) for mprm in range(-nprm,nprm+1) ]
    pvals = [ p for p in range(1,p_max_main+1) for _ in range(-p,p+1) ]
    qvals = [ q for p in range(1,p_max_main+1) for q in range(-p,p+1) ]
    nvals = [ n for n in range(1,n_max_main+1) for _ in range(-n,n+1) ]
    mvals = [ m for n in range(1,n_max_main+1) for m in range(-n,n+1) ]
    Nnmprm = len(nprmvals)
    Npq = len(pvals)
    Nnm = len(nvals)

    # Define scaling for specifying distances in terms of body radii
    # Note: r_io MUST be negative
    if bname is None:
        rscale_moments = 1.0
        rscale_asym = 1.0
        r_io = -3
    elif bname == "Callisto":
        rscale_moments = 1/2410.3/1e3
        rscale_asym = 1/2510300
        r_io = -1
    elif bname == "Enceladus":
        rscale_moments = 1/252.1/1e3
        rscale_asym = 1/230035.1
        r_io = -2
    elif bname == "Europa":
        r_io = -2
        if compare_me:
            rscale_moments = 1/1560.0/1e3
            rscale_asym = 1/1537500
        else:
            rscale_moments = 1/1561.0/1e3
            if seawater:
                rscale_asym = 1/1538503.78421254
            else:
                rscale_asym = 1/1538489.72081887
    elif bname == "Miranda":
        rscale_moments = 1/235.8/1e3
        rscale_asym = 1/185989.764241229
        r_io = -3
    elif bname == "Triton":
        rscale_moments = 1/1353.4/1e3
        rscale_asym = 1/1803400
        r_io = -1

    R = 1 / rscale_moments / 1e3

    if debug:
        print("Debug: Getting Xi/Xid values")
        Xid = asym.get_all_Xid(nprm_max_main, p_max_main, n_max_main, nvals, mvals)
        print("#######   Xid values   #######")
        asym.print_Xid_table(Xid, nprm_max_main, p_max_main, n_max_main)
        print(" ")
        print("#######   Xi values   #######")
        asym.print_Xi_table(nprm_max_main, p_max_main, n_max_main)

    if recalc:
        int_model = inp_path + "interior_model_asym" + bfname + bname_opt + sw_opt + ".txt"
        print("Using interior model: " + int_model)

        r_bds, sigmas, bcdev = np.loadtxt(int_model, skiprows=1, unpack=True, delimiter=',')
        n_bds = np.size(r_bds)

        # Read in asymmetric shape information
        eps_scaled = r_bds * bcdev
        if relative:
            single_asym = None
        else:
            single_asym = r_io
        asym_shape, grav_shape = asym.read_shape(n_bds, p_max_main, rscale_asym, bodyname=bname, relative=relative, single_asym=single_asym,
                                     eps_scaled=eps_scaled, r_bds=r_bds, r_io=r_io, append=bname_opt, convert_depth_to_chipq=convert_depth_to_chipq)
        r_bds, sigmas, asym_shape = asym.validate(r_bds, sigmas, bcdev, asym_shape, p_max_main)

        # Read in Benm info
        if plot_field:
            peak_periods, Benm = asym.read_Benm(nprm_max_main, p_max_main, bodyname=bname, synodic=synodic_only, orbital=False)
            peak_omegas = 2*np.pi/(peak_periods*3600)
            if not isinstance(peak_omegas, Iterable):
                peak_periods = [peak_periods]
                peak_omegas = [peak_omegas]
            n_peaks = len(peak_omegas)

    else:
        r_bds = None
        asym_shape = None
        grav_shape = None

    if debug:
        fpath = "interior/Avals.dat"
        if recalc:
            Xid = None
            rscaling = None
            omegas = np.linspace(1e-3, 100, 250)  # These will actually be used as kr values
            Binm, Aes, Ats, Ads, krvals = asym.BiList(r_bds, sigmas, omegas, asym_shape, grav_shape, Benm, rscale_moments, nvals, mvals, p_max_main, writeout=False, nprm_max=nprm_max_main, bodyname=bname, debug=True)

            fout = open(fpath, "w")
            header = "{:<24}, {:<24}, {:<24}, {:<24}, {:<24}, {:<24}, {:<24}, {:<24}, {:<24}\n".format("|kr|,", "|A_1^e|,", "-arg(A_1^e),", "|A_1^t|,", "-arg(A_1^t),", "|A_1^\star|", "-arg(A_1^\star)", "|A_1^tA_1^\star|", "-arg(A_1^tA_1^\star)")
            fout.write(header)

            AtAd = Ats[:, 0]*Ads[:, 0]
            kr = [np.abs(krval) for krval in krvals]
            Ae_mag = [np.abs(val) for val in Aes[:, 0]]
            Ae_arg = [-np.angle(val) for val in Aes[:, 0]]
            At_mag = [np.abs(val) for val in Ats[:, 0]]
            At_arg = [-np.angle(val) for val in Ats[:, 0]]
            Ad_mag = [np.abs(val) for val in Ads[:, 0]]
            Ad_arg = [-np.angle(val) for val in Ads[:, 0]]
            AtAd_mag = [np.abs(val) for val in AtAd]
            AtAd_arg = [-np.angle(val) for val in AtAd]

            for i in range(len(omegas)):
                this_line = str(kr[i]) + ", " + str(Ae_mag[i]) + ", " + str(Ae_arg[i]) + ", " + str(At_mag[i]) + ", " + str(At_arg[i]) + ", " + str(Ad_mag[i]) + ", " + str(Ad_arg[i]) + ", " + str(AtAd_mag[i]) + ", " + str(AtAd_arg[i]) + "\n"
                fout.write(this_line)
            fout.close()
            print("Data for A functions written to file: ", fpath)
        else:
            kr, Ae_mag, Ae_arg, At_mag, At_arg, Ad_mag, Ad_arg, AtAd_mag, AtAd_arg = np.loadtxt(fpath, skiprows=1, unpack=True, delimiter=',')

        plots.plotAfunctions(kr, 1, Ae_mag, Ae_arg, At_mag, At_arg, Ad_mag, Ad_arg, AtAd_mag, AtAd_arg)
        print("Debug: quitting")
        quit()

    if plot_asym:
        if bname == "Triton" or bname == "Callisto":
            R_surface = -2
            r_io = -1
            descrip = "Ionosphere"
        else:
            R_surface = r_io + 1
            if do_large:
                descrip = "Ice shell"
            else:
                descrip = "Ice--ocean"
        plots.plotAsym(recalc, do_large, index=r_io, cmp_index=R_surface, r_bds=r_bds, asym_shape=asym_shape, pvals=pvals, qvals=qvals, bodyname=bname, append=bname_opt+sw_opt, descrip=descrip, no_title=no_title_text)

    if plot_field:
        if recalc:
            # Calculate and print to data files
            Binm_sph = sym.BiList(r_bds, sigmas, peak_omegas, Benm, nprmvals, mprmvals, rscale_moments, n_max=nprm_max_main, bodyname=bname, append=bname_opt+sw_opt)
            Binm = asym.BiList(r_bds, sigmas, peak_omegas, asym_shape, grav_shape, Benm, rscale_moments, nvals, mvals, p_max_main, nprm_max=nprm_max_main, bodyname=bname, append=bname_opt+sw_opt)
        else:
            T_hrs, n_asy, m_asy, lin_Binm_Re, lin_Binm_Im = np.loadtxt(inp_Bi_path+bin_name+"Binm_asym"+bname_opt+sw_opt+".dat", skiprows=1, unpack=True, delimiter=',')
            peak_periods = np.unique(T_hrs)
            n_peaks = len(peak_periods)
            Nnm_asy = int(len(n_asy)/n_peaks)
            nmax_asy = int(np.sqrt(Nnm_asy + 1)) - 1
            lin_Binm = lin_Binm_Re + 1j*lin_Binm_Im
            Binm = np.reshape(lin_Binm, (n_peaks,Nnm_asy))

            T_hrs_sym, n_sph, m_sph, lin_Binm_sph_Re, lin_Binm_sph_Im = np.loadtxt(inp_Bi_path+bin_name+"Binm_sym"+bname_opt+sw_opt+".dat", skiprows=1, unpack=True, delimiter=',')
            peak_periods_sym = np.unique(T_hrs_sym)
            n_peaks_sym = len(peak_periods_sym)
            try:
                same_spectrum = (np.equal(peak_periods, peak_periods_sym)).all()
            except:
                raise ValueError("Periods of excitation for Binm_sph and Binm did not match. Set recalc=True and run again to fix the problem.")
            if not same_spectrum:
                raise ValueError("Periods of excitation for Binm_sph and Binm did not match. Set recalc=True and run again to fix the problem.")
            Nnm_sph = int(len(n_sph)/n_peaks)
            nmax_sph = int(np.sqrt(Nnm_sph + 1)) - 1
            lin_Binm_sph = lin_Binm_sph_Re + 1j*lin_Binm_sph_Im
            Binm_sph = np.reshape(lin_Binm_sph, (n_peaks,Nnm_sph))
            peak_omegas = 2*np.pi/(peak_periods*3600)

        plot_on_sphere = True
        if plot_on_sphere:
            asym_frac = None
        else:
            asym_frac = asym_shape[r_io,...] / r_bds[r_io]

        Binm_sph_rot = Binm_sph*1.0
        Binm_rot = Binm*1.0
        if do_gif:
            print("Making "+str(n_frames)+" animation frames.")
            for iT in range(n_frames):
                iT_str = f'{iT:04}'
                t_hr = peak_periods[-1] * iT / n_frames
                tstr = f'{round(t_hr, 1):03}'
                tframe = tsec + t_hr*3600
                for i_om in range(n_peaks):
                    Binm_sph_rot[i_om,...] = Binm_sph[i_om,...] * np.exp(-1j * peak_omegas[i_om] * tframe)
                    Binm_rot[i_om,...]     = Binm[i_om,...]     * np.exp(-1j * peak_omegas[i_om] * tframe)

                plots.plotMagSurf(n_peaks, Binm_rot, nvals, mvals, do_large, r_surf_mean=eval_r, asym_frac=asym_frac, pvals=pvals, qvals=qvals,
                                  difference=gif_diff, Binm_sph=Binm_sph_rot, nprmvals=nprmvals, mprmvals=mprmvals, bodyname=bname,
                                  append=bname_opt+sw_opt, fend=iT_str, tstr=tstr, component=comp, no_title=False)

            print("Animation frames printed to figures/anim_frames/ folder.")
            print("Stack them into a gif with, e.g.:")
            print("convert -delay 15 figures/anim_frames/Miranda_field_asym0*.png -loop 15 figures/anim_Miranda_asym.gif")
        else:
            for i_om in range(n_peaks):
                Binm_sph_rot[i_om,...] = Binm_sph[i_om,...] * np.exp(-1j * peak_omegas[i_om] * tsec)
                Binm_rot[i_om,...]     = Binm[i_om,...]     * np.exp(-1j * peak_omegas[i_om] * tsec)

            # Plot symmetric moments/difference
            plots.plotMagSurf(n_peaks, Binm_rot, nvals, mvals, do_large, r_surf_mean=eval_r, asym_frac=asym_frac, pvals=pvals, qvals=qvals, difference=True,
                            Binm_sph=Binm_sph_rot, nprmvals=nprmvals, mprmvals=mprmvals, bodyname=bname,
                            append=bname_opt+sw_opt, component=comp, absolute=absolute, no_title=no_title_text)

        # Restrict plotting of certain diagnostic plots so we don't get spammed when we only want to do this for special conditions
        actually_plot_traces = bname == "Europa" and (compare_me or seawater) and comp == "x"
        if sub_planet_vert and actually_plot_traces:
            if not recalc:
                peak_periods, Benm = asym.read_Benm(nprm_max_main, p_max_main, bodyname=bname, synodic=synodic_only)
                int_model = inp_path + "interior_model_asym" + bfname + bname_opt + sw_opt + ".txt"
                r_bds, sigmas, bcdev = np.loadtxt(int_model, skiprows=1, unpack=True, delimiter=',')

            t_cut = vert_cut_hr * 3600
            vert_begin = np.maximum(vert_start, r_bds[-1]*rscale_moments)
            r = np.linspace(vert_begin, vert_stop, t_pts)
            x = r * np.cos(np.radians(vert_cut_lat)) * np.cos(np.radians(vert_cut_lon))
            y = r * np.cos(np.radians(vert_cut_lat)) * np.sin(np.radians(vert_cut_lon))
            z = r * np.sin(np.radians(vert_cut_lat))
            t = np.zeros(t_pts)
            t += t_cut
            plots.plotTrajec(x,y,z,r,t, Binm, Benm, peak_omegas, nprm_max_main, n_max_main, nvals, mvals, R_body=R, component=comp, difference=True, Binm_sph=Binm_sph, bodyname=bname, append=bname_opt+sw_opt+compstr)

        # Plot a time series at the sub-parent-planet point--(0°, 0°) in IAU coordinates.
        # Only tested for Europa, with no ionosphere.
        if synodic_only and actually_plot_traces:
            if not recalc:
                synodic_period, Benm = asym.read_Benm(nprm_max_main, p_max_main, bodyname=bname, synodic=True)
                peak_periods = [synodic_period]
            if not sub_planet_vert:
                int_model = inp_path + "interior_model_asym" + bfname + bname_opt + sw_opt + ".txt"
                r_bds, sigmas, bcdev = np.loadtxt(int_model, skiprows=1, unpack=True, delimiter=',')

            r = (localt + R) / R

            # Only valid if the evaluation point is outside the conducting region, so that B is subject to the Laplace equation.
            if r*R*1e3/r_bds[-1] >= 0.999:
                locx = r * np.cos(np.radians(loclat)) * np.cos(np.radians(loclon))
                locy = r * np.cos(np.radians(loclat)) * np.sin(np.radians(loclon))
                locz = r * np.sin(np.radians(loclat))
                loc = [ locx, locy, locz, r ] # Now IAU planetocentric in terms of body radii
                plots.plotTimeSeries(loc, Binm[0,...], Benm[0,...], tsec, peak_periods[0], nprm_max_main, n_max_main, nvals, mvals,
                                     n_pts=t_pts, component=comp, Binm_sph=Binm_sph[0,...], bodyname=bname, append=bname_opt+sw_opt+compstr)

                if orbital_time_series:
                    if not recalc:
                        n_bds = np.size(r_bds)
                        eps_scaled = r_bds * bcdev
                        if relative:
                            single_asym = None
                        else:
                            single_asym = r_io
                        asym_shape, grav_shape = asym.read_shape(n_bds, p_max_main, rscale_asym, bodyname=bname, relative=relative, single_asym=single_asym,
                                                     eps_scaled=eps_scaled, r_bds=r_bds, r_io=r_io, append=bname_opt, convert_depth_to_chipq=convert_depth_to_chipq)
                        r_bds, sigmas, asym_shape = asym.validate(r_bds, sigmas, bcdev, asym_shape, p_max_main)

                    orbital_period, Benm_orbital = asym.read_Benm(nprm_max_main, p_max_main, bodyname=bname, orbital=True)
                    orbital_omega = 2*np.pi/(orbital_period*3600)
                    Binm_sph_orbital = sym.BiList(r_bds, sigmas, [orbital_omega], Benm_orbital, nprmvals, mprmvals, rscale_moments, n_max=nprm_max_main, bodyname=bname, append=bname_opt + sw_opt)
                    Binm_orbital = asym.BiList(r_bds, sigmas, [orbital_omega], asym_shape, grav_shape, Benm_orbital, rscale_moments, nvals, mvals, p_max_main, nprm_max=nprm_max_main, bodyname=bname, append=bname_opt + sw_opt)
                    plots.plotTimeSeries(loc, Binm_orbital[0, ...], Benm_orbital[0, ...], tsec, orbital_period, nprm_max_main, n_max_main, nvals, mvals, n_pts=t_pts, component=comp, Binm_sph=Binm_sph_orbital[0, ...], bodyname=bname, append=bname_opt+sw_opt+compstr+"_orbital")