""" This program runs all calculations for induced magnetic fields
    from near-spherical conductors and plots results that appear in
    the associated publication.
    Developed in Python 3.8 for "A perturbation method for evaluating the
    magnetic field induced from an arbitrary, asymmetric ocean world 
    analytically" by Styczinski et al.
    DOI: 10.1016/j.icarus.2021.114840
Author: M. J. Styczinski, mjstyczi@uw.edu """

import MoonMag.eval_induced_field as eval

def run_all():
    # Each body to include in calculations
    do_Europa = True
    do_Miranda = True
    do_Callisto = True
    do_Triton = True
    
    # Make gifs of the specified component for each body after the calculations are done
    make_gifs = False
    gif_comp = None
    
    # Settings
    do_recalc = True  # Whether to recalculate the induced moments and contours when they will be plotted (only affects first run for each body, as the moments are the same for each component)
    initial_contour = False  # Whether to plot a contour map along with field magnitudes (the first calc run for each body). Warning: this takes a long time at high resolution!
    do_fields = True  # Whether to evaluate and plot fields
    do_detailed = True  # Whether to make standard plots (this option is independent from do_large_plots)
    do_large_plots = False  # Whether to make copies with larger print and no colorbars, for cramped spaces
    
    # Uses the following command structure:
    # eval.run_calcs(bname, comp, recalc, plot_field, plot_asym, do_large=False, seawater=False, compare_me=False, do_gif=False)

    if do_Miranda:
        bname = "Miranda"
        print(f" - {bname} - ")
        if do_detailed:
            eval.run_calcs(bname, None, do_recalc, do_fields, initial_contour, do_large=False)
            eval.run_calcs(bname, "x", False, do_fields, False, do_large=False)
            eval.run_calcs(bname, "y", False, do_fields, False, do_large=False)
            eval.run_calcs(bname, "z", False, do_fields, False, do_large=False)

        if do_large_plots:
            eval.run_calcs(bname, None, False, do_fields, initial_contour, do_large=True)
            eval.run_calcs(bname, "x", False, do_fields, False, do_large=True)
            eval.run_calcs(bname, "y", False, do_fields, False, do_large=True)
            eval.run_calcs(bname, "z", False, do_fields, False, do_large=True)

    if do_Europa:
        bname = "Europa"
        print(f" - {bname} Tobie model high salinity (Seawater) - ")
        if do_detailed:
            eval.run_calcs(bname, None, do_recalc, do_fields, initial_contour, seawater=True)
            eval.run_calcs(bname, "x", False, do_fields, False, seawater=True)
            eval.run_calcs(bname, "y", False, do_fields, False, seawater=True)
            eval.run_calcs(bname, "z", False, do_fields, False, seawater=True)

        if do_large_plots:
            eval.run_calcs(bname, None, False, do_fields, initial_contour, do_large=True, seawater=True)
            eval.run_calcs(bname, "x", False, do_fields, False, do_large=True, seawater=True)
            eval.run_calcs(bname, "y", False, do_fields, False, do_large=True, seawater=True)
            eval.run_calcs(bname, "z", False, do_fields, False, do_large=True, seawater=True)
    
        bname = "Europa"
        print(f" - {bname} Tobie model low salinity (10% Seawater) - ")
        if do_detailed:
            eval.run_calcs(bname, None, do_recalc, do_fields, initial_contour, seawater=False)
            eval.run_calcs(bname, "x", False, do_fields, False, seawater=False)
            eval.run_calcs(bname, "y", False, do_fields, False, seawater=False)
            eval.run_calcs(bname, "z", False, do_fields, False, seawater=False)
    
        if do_large_plots:
            eval.run_calcs(bname, None, False, do_fields, initial_contour, do_large=True, seawater=False)
            eval.run_calcs(bname, "x", False, do_fields, False, do_large=True, seawater=False)
            eval.run_calcs(bname, "y", False, do_fields, False, do_large=True, seawater=False)
            eval.run_calcs(bname, "z", False, do_fields, False, do_large=True, seawater=False)
    
        bname = "Europa"
        print(f" - {bname} previous comparison - ")
        if do_detailed:
            eval.run_calcs(bname, None, do_recalc, do_fields, initial_contour, compare_me=True)
            eval.run_calcs(bname, "x", False, do_fields, False, compare_me=True)
            eval.run_calcs(bname, "y", False, do_fields, False, compare_me=True)
            eval.run_calcs(bname, "z", False, do_fields, False, compare_me=True)
    
        if do_large_plots:
            eval.run_calcs(bname, None, False, do_fields, initial_contour, do_large=True, compare_me=True)
            eval.run_calcs(bname, "x", False, do_fields, False, do_large=True, compare_me=True)
            eval.run_calcs(bname, "y", False, do_fields, False, do_large=True, compare_me=True)
            eval.run_calcs(bname, "z", False, do_fields, False, do_large=True, compare_me=True)
    
    if do_Callisto:
        bname = "Callisto"
        print(f" - {bname} - ")
        if do_detailed:
            eval.run_calcs(bname, None, do_recalc, do_fields, initial_contour)
            eval.run_calcs(bname, "x", False, do_fields, False)
            eval.run_calcs(bname, "y", False, do_fields, False)
            eval.run_calcs(bname, "z", False, do_fields, False)
    
        if do_large_plots:
            eval.run_calcs(bname, None, False, do_fields, initial_contour, do_large=True)
            eval.run_calcs(bname, "x", False, do_fields, False, do_large=True)
            eval.run_calcs(bname, "y", False, do_fields, False, do_large=True)
            eval.run_calcs(bname, "z", False, do_fields, False, do_large=True)
    
    if do_Triton:
        bname = "Triton"
        print(f" - {bname} - ")
        if do_detailed:
            eval.run_calcs(bname, None, do_recalc, do_fields, initial_contour)
            eval.run_calcs(bname, "x", False, do_fields, False)
            eval.run_calcs(bname, "y", False, do_fields, False)
            eval.run_calcs(bname, "z", False, do_fields, False)
    
        if do_large_plots:
            eval.run_calcs(bname, None, False, do_fields, initial_contour, do_large=True)
            eval.run_calcs(bname, "x", False, do_fields, False, do_large=True)
            eval.run_calcs(bname, "y", False, do_fields, False, do_large=True)
            eval.run_calcs(bname, "z", False, do_fields, False, do_large=True)
    
    if make_gifs:
        if do_Europa:
            eval.run_calcs("Europa", gif_comp, False, True, False, seawater=True, do_gif=True)
            eval.run_calcs("Europa", gif_comp, False, True, False, seawater=False, do_gif=True)
            eval.run_calcs("Europa", gif_comp, False, True, False, compare_me=True, do_gif=True)
        if do_Miranda:
            eval.run_calcs("Miranda", gif_comp, False, True, False, do_gif=True)
        if do_Callisto:
            eval.run_calcs("Callisto", gif_comp, False, True, False, do_gif=True)
        if do_Triton:
            eval.run_calcs("Triton", gif_comp, False, True, False, do_gif=True)


if __name__ == "__main__":
    run_all()
