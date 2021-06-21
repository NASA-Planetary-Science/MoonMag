# Supplemental Materials for "An analytic solution for evaluating the magnetic field induced from an arbitrary, asymmetric ocean world"
Python software for evaluating magnetic fields from asymmetric ocean moons. Written in Python 3.8. For help, please contact M. Styczinski at [mjstyczi@uw.edu](mailto:mjstyczi@uw.edu).

## Quick start
* Run the file `run_all.py` using a Python 3 interpreter.

You will need the following Python modules:
* numpy
* matplotlib
* skyfield
* mpmath
* scipy
* cartopy
* multiprocessing

To generate contour plots showing the asymmetric layer geometry, you will need to edit line 23 in `run_all.py`. These plots take a long time to generate, so they are turned off by default. Animations are set to plot differences in the magnitudes by default. Change this by setting `gif_comp = "x"`, `"y"`, or `"z"` at line 19 in the same file.

## Configuration
Run settings are split between `run_all.py`, `config.py`, `eval_induced_field.py`, and `plotting_funcs.py`.
* `run_all.py`: Set which bodies to run, whether to plot animation frames, etc.
* `config.py`: Set plot resolutions, precision for calculations, altitudes for magnetic field maps, date/time for evaluation, etc.
* `eval_induced_field.py`: Set body radii, maximum shape harmonic degree p, and other normalizations.
* `plotting_funcs.py`: Plot labels, image types, and contour/colormap settings are found within individual plotting functions.

## Editing interior models
Interior models are generated using [PlanetProfile](https://github.com/vancesteven/PlanetProfile). The specific PlanetProfile run files (called `PP<Body>_asymmetry.m`) used to create the interior conductivity profiles used in this work are contained in the `interior/PP_models/` directory. A modified version of the main PlanetProfile function is contained in `interior/PP_models/PlanetProfile_snapshot.m`. A future version of PlanetProfile will incorporate the changes contained in this file, which are primarily to interpolate the detailed ocean conductivity profile down to just 3 layers (and condense nonconducting layers).

To create a conductivity profile, run a `PP<Body>_asymmetry.m` file from the appropriate (`<Body>/`) directory in your PlanetProfile installation, with `PlanetProfile_snapshot.m` replacing the main `PlanetProfile.m` file. Then, edit the final value in each row of the printed file `interior_model_asym_<Body>.txt` to match the desired asymmetry. These values are ε<sub>l</sub>/r̅<sub>l</sub> as discussed in Section 4.1.2 of the paper. Last, move the edited file to the `interior/` directory here.

### Creating asymmetric shape files
1. Collect real-valued, Schmidt semi-normalized harmonic coefficients into a single file in interior/ named `depth_chi_pq_shape_<Body>.txt`.
   * See the corresponding file for Miranda for an example. These values are assumed to be layer thicknesses, i.e. depths from the surface to the top of an ocean.
   * The default settings in `eval_induced_field.py` apply these coefficients to the layer with index `r_io`, which is negative. For bodies with no ionosphere, typically `r_io` is -2, the second-to-top layer.
1. In config.py, set `relative = False` and `convert_depth_to_chipq = True` and run a field calculation (e.g. `python run_all.py` at the terminal). This prints a new file, `interior/chi_pq_<Body>.txt`.
   * The results of this calculation will only apply asymmetry to the layer corresponding to `r_io`.
1. Open the new file and the corresponding `interior_model_asym_<Body>.txt` file. For each concentric ocean layer, set the ε<sub>l</sub>/r̅<sub>l</sub> values to be the bcdev value reported at the end of the header line of `chi_pq_<Body>.txt` divided by the radius of the layer corresponding to `r_io`.
   * For example, for Miranda `bcdev = 39120.39 m` and `r_io = -3` as it has an ionosphere. The radius of that boundary is 185989.764 m, so the ε<sub>l</sub>/r̅<sub>l</sub> value to use is 39120.39 / 185989.764 = 0.21. Each layer should have the same value here if they are concentric--the absolute asymmetry is scaled to the radius of the boundary.
1. For each degree p up to the desired pmax, create a file in `interior/` named `degree<p>_shapes_<Body>.txt` and add a single descriptive header line to each.
   * See the corresponding Miranda files for examples.
1. Copy the lines of the `chi_pq_<Body>.txt` file for each harmonic *multiple times* into the corresponding `degree<p>_shapes` files, with one line for each layer in `interior_model_asym_<Body>.txt`.
   * If desired, these coefficients (the χ<sup>l</sup><sub>pq</sub> values) may be adjusted to specify asymmetric layers independently.
1. In `config.py`, set `relative = True`. Executing `run_all.py` will now use the values in `degree<p>_shapes` and `interior_model_asym` files to set the asymmetry for each layer.
