""" This program contains a function for calculating magnetic fields
    in planetocentric Cartesian coordinates from spherical magnetic moments.
    Developed in Python 3.8 for "A perturbation method for evaluating the
    magnetic field induced from an arbitrary, asymmetric ocean world 
    analytically" by Styczinski et al.
    DOI: 10.1016/j.icarus.2021.114840
Author: M. J. Styczinski, mjstyczi@uw.edu """

import numpy as np
import scipy as sci
from math import *

"""
eval_Be()
    Evaluates the excitation field at the given coordinates due to a particular magnetic moment Benm.
    Usage: `Bx`, `By`, `Bz` = eval_Be(`n`, `m`, `Benm`, `x`, `y`, `z`, `r`, `omega=None`, `t=None`)
    Returns:
        Bx, By, Bz (each): complex, ndarray shape(Nvals). A linear array of field values due to these particular n,m values,
            at each of the specified points. Returns a time sequence if ω and t are passed.
    Parameters:
        n: integer. Degree of magnetic moment to be evaluated.
        m: integer. Order of magnetic moment to be evaluated.
        Benm: complex. Excitation moment of degree and order n,m. Units match the output field.
        x,y,z,r: float, shape(Nvals). Linear arrays of corresponding x,y, and z values. r is not needed
            but requiring it as an argument makes the function call identical to eval_Bi. If omega and t
            are passed, these quantities are the trajectory locations.
        omega: float (None). Optional oscillation frequency in rads/s for evaluating time series. Requires t to be passed as well.
        t: float, shape(Nvals) (None). Optional time values in TDB seconds since J2000 epoch. Required if omega is passed.
    """
def eval_Be(n,m,Benm, x,y,z,r, omega=None, t=None):

    if omega is None:
        timeRot = 1.0
    else:
        timeRot = np.exp(-1j * omega * t)

    if n == 1:
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #   Uniform field components
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        A1 = 1/2*np.sqrt(3/2/np.pi)
        B_base = A1*Benm

        if m == -1:
            Bx = -B_base
            By = 1j*B_base
            Bz = 0
        elif m == 0:
            Bx = 0
            By = 0
            Bz = -sqrt(2)*B_base
        elif m == 1:
            Bx = B_base
            By = 1j*B_base
            Bz = 0
        else:
            print(" m = ", m)
            raise ValueError("In field_xyz.eval_Be, n=1 and m is not between -n and n.")

    elif n==2:
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #   Linear field components
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        A2 = -1/4*np.sqrt(15/2/np.pi)
        B_base = -2*A2*Benm

        rt2o3 = sqrt(2/3)

        if m==-2:
            Bx = B_base * (-x + 1j*y)
            By = B_base * (y + 1j*x)
            Bz = 0
        elif m==-1:
            Bx = B_base * -z
            By = B_base * 1j*z
            Bz = B_base * (-x + 1j*y)
        elif m==0:
            Bx = B_base * rt2o3*x
            By = B_base * rt2o3*y
            Bz = B_base * -2*rt2o3*z
        elif m==1:
            Bx = B_base * z
            By = B_base * 1j*z
            Bz = B_base * (x + 1j*y)
        elif m==2:
            Bx = B_base * (-x - 1j*y)
            By = B_base * (y - 1j*x)
            Bz = 0
        else:
            print(" m = ", m)
            raise ValueError("In field_xyz.eval_Bi, n=2 and m is not between -n and n.")

    else:
        print(" n = ", n)
        raise ValueError("In field_xyz.eval_Be, n>2 but only n=1 to n=2 are supported.")

    Bx = Bx * timeRot
    By = By * timeRot
    Bz = Bz * timeRot

    return Bx, By, Bz
    
#############################################

"""
eval_Bi()
    Evaluates the induced magnetic field at the given coordinates due to a particular magnetic moment Binm.
    Usage: `Bx`, `By`, `Bz` = eval_Bi(`n`, `m`, `Binm`, `x`, `y`, `z`, `r`, `omega=None`, `t=None`)
    Returns:
        Bx, By, Bz (each): complex, ndarray shape(Nvals). A linear array of field values due to these particular n,m values,
            at each of the specified points. Returns a time sequence if ω and t are passed.
    Parameters:
        n: integer. Degree of magnetic moment to be evaluated.
        m: integer. Order of magnetic moment to be evaluated.
        Binm: complex. Magnetic moment of degree and order n,m. Units match the output field.
        x,y,z,r: float, shape(Nvals). Linear arrays of corresponding x,y, and z values. r is redundant but
            saves on computation time to avoid recalculating on every call to this function. If omega and t
            are passed, these quantities are the trajectory locations.
        omega: float (None). Optional oscillation frequency in rads/s for evaluating time series. Requires t to be passed as well.
        t: float, shape(Nvals) (None). Optional time values in TDB seconds since J2000 epoch. Required if omega is passed.
    """
def eval_Bi(n,m,Binm, x,y,z,r, omega=None, t=None):

    if omega is None:
        timeRot = 1.0
    else:
        timeRot = np.exp(-1j*omega*t)

    if n==1:
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #   Dipole field components
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        A1 = 1/2*np.sqrt(3/2/np.pi)
        Bx = A1*Binm/r**5
        By = Bx + 0.0
        Bz = Bx + 0.0

        rt2 = sqrt(2)
        if m==-1:
            Bx = Bx * ( (2*x**2 - y**2 - z**2) + 1j*(-3*x*y) )
            By = By * ( (3*x*y) + 1j*(x**2 - 2*y**2 + z**2) )
            Bz = Bz * ( (3*x*z) + 1j*(-3*y*z) )
        elif m==0:
            Bx = Bx * rt2*(3*x*z)
            By = By * rt2*(3*y*z)
            Bz = Bz * rt2*(-x**2 - y**2 + 2*z**2)
        elif m==1:
            Bx = Bx * ( (-2*x**2 + y**2 + z**2) + 1j*(-3*x*y) )
            By = By * ( (-3*x*y) + 1j*(x**2 - 2*y**2 + z**2) )
            Bz = Bz * ( (-3*x*z) + 1j*(-3*y*z) )
        else:
            print(" m = ", m)
            raise ValueError("In field_xyz.eval_Bi, n=1 and m is not between -n and n.")

    elif n==2:
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #   Quadrupole moment components
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        A2 = -1/4*np.sqrt(15/2/np.pi)
        Bx = A2*Binm/r**7
        By = Bx + 0.0
        Bz = Bx + 0.0

        rt2o3 = sqrt(2/3)

        if m==-2:
            Bx = Bx * ( (-3*x**3 + 7*x*y**2 + 2*x*z**2) + 1j*(8*x**2*y - 2*y**3 - 2*y*z**2) )
            By = By * ( (-7*x**2*y + 3*y**3 - 2*y*z**2) + 1j*(-2*x**3 + 8*x*y**2 - 2*x*z**2) )
            Bz = Bz * ( (-5*x**2*z + 5*y**2*z) + 1j*(10*x*y*z) )
        elif m==-1:
            Bx = Bx * 2*( (-4*x**2*z + y**2*z + z**3) + 1j*(5*x*y*z) )
            By = By * 2*( (-5*x*y*z) + 1j*(-x**2*z + 4*y**2*z - z**3) )
            Bz = Bz * 2*( (x**3 + x*y**2 - 4*x*z**2) + 1j*(-x**2*y - y**3 + 4*y*z**2) )
        elif m==0:
            Bx = Bx * rt2o3*(3*x**3 + 3*x*y**2 - 12*x*z**2)
            By = By * rt2o3*(3*x**2*y + 3*y**3 - 12*y*z**2)
            Bz = Bz * rt2o3*(9*x**2*z + 9*y**2*z - 6*z**3)
        elif m==1:
            Bx = Bx * -2*( (-4*x**2*z + y**2*z + z**3) + 1j*(-5*x*y*z) )
            By = By * -2*( (-5*x*y*z) + 1j*(x**2*z - 4*y**2*z + z**3) )
            Bz = Bz * -2*( (x**3 + x*y**2 - 4*x*z**2) + 1j*(x**2*y + y**3 - 4*y*z**2) )
        elif m==2:
            Bx = Bx * ( (-3*x**3 + 7*x*y**2 + 2*x*z**2) + 1j*(-8*x**2*y + 2*y**3 + 2*y*z**2) )
            By = By * ( (-7*x**2*y + 3*y**3 - 2*y*z**2) + 1j*(2*x**3 - 8*x*y**2 + 2*x*z**2) )
            Bz = Bz * ( (-5*x**2*z + 5*y**2*z) + 1j*(-10*x*y*z) )
        else:
            print(" m = ", m)
            raise ValueError("In field_xyz.eval_Bi, n=2 and m is not between -n and n.")

    elif n==3:
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #   Octupole moment components
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        A3 = 1/8*np.sqrt(35/np.pi)
        Bx = A3*Binm/r**9
        By = Bx + 0.0
        Bz = Bx + 0.0

        rt6 = sqrt(6)
        rt3o5 = sqrt(3/5)
        rt4o5 = sqrt(4/5)

        if m==-3:
            Bx = Bx * ( (4*x**4 - 21*x**2*y**2 - 3*x**2*z**2 + 3*y**4 + 3*y**2*z**2) + 1j*(-15*x**3*y + 13*x*y**3 + 6*x*y*z**2) )
            By = By * ( (13*x**3*y - 15*x*y**3 + 6*x*y*z**2) + 1j*(3*x**4 - 21*x**2*y**2 + 3*x**2*z**2 + 4*y**4 - 3*y**2*z**2) )
            Bz = Bz * ( (7*x**3*z - 21*x*y**2*z) + 1j*(-21*x**2*y*z + 7*y**3*z) )
        elif m==-2:
            Bx = Bx * rt6*( (5*x**3*z - 9*x*y**2*z - 2*x*z**3) + 1j*(-12*x**2*y*z + 2*y**3*z + 2*y*z**3) )
            By = By * rt6*( (9*x**2*y*z - 5*y**3*z + 2*y*z**3) + 1j*(2*x**3*z - 12*x*y**2*z + 2*x*z**3) )
            Bz = Bz * rt6*( (-x**4 + 6*x**2*z**2 + y**4 - 6*y**2*z**2) + 1j*(2*x**3*y + 2*x*y**3 - 12*x*y*z**2) )
        elif m==-1:
            Bx = Bx * rt3o5*( (-4*x**4 -3*x**2*y**2 + 27*x**2*z**2 + y**4 - 3*y**2*z**2 - 4*z**4) + 1j*(5*x**3*y + 5*x*y**3 - 30*x*y*z**2) )
            By = By * rt3o5*( (-5*x**3*y - 5*x*y**3 + 30*x*y*z**2) + 1j*(-x**4 + 3*x**2*y**2 + 3*x**2*z**2 + 4*y**4 - 27*y**2*z**2 + 4*z**4) )
            Bz = Bz * rt3o5*( (-15*x**3*z - 15*x*y**2*z + 20*x*z**3) + 1j*(15*x**2*y*z + 15*y**3*z - 20*y*z**3) )
        elif m==0:
            Bx = Bx * rt4o5*(-15*x**3*z - 15*x*y**2*z + 20*x*z**3)
            By = By * rt4o5*(-15*x**2*y*z - 15*y**3*z + 20*y*z**3)
            Bz = Bz * rt4o5*(3*x**4 + 6*x**2*y**2 - 24*x**2*z**2 + 3*y**4 - 24*y**2*z**2 + 8*z**4)
        elif m==1:
            Bx = Bx * -rt3o5*( (-4*x**4 - 3*x**2*y**2 + 27*x**2*z**2 + y**4 - 3*y**2*z**2 - 4*z**4) + 1j*(-5*x**3*y - 5*x*y**3 + 30*x*y*z**2) )
            By = By * -rt3o5*( (-5*x**3*y - 5*x*y**3 + 30*x*y*z**2) + 1j*(x**4 - 3*x**2*y**2 - 3*x**2*z**2 - 4*y**4 + 27*y**2*z**2 - 4*z**4) )
            Bz = Bz * -rt3o5*( (-15*x**3*z - 15*x*y**2*z + 20*x*z**3) + 1j*(-15*x**2*y*z - 15*y**3*z + 20*y*z**3) )
        elif m==2:
            Bx = Bx * rt6*( (5*x**3*z - 9*x*y**2*z - 2*x*z**3) + 1j*(12*x**2*y*z - 2*y**3*z - 2*y*z**3) )
            By = By * rt6*( (9*x**2*y*z - 5*y**3*z + 2*y*z**3) + 1j*(-2*x**3*z + 12*x*y**2*z - 2*x*z**3) )
            Bz = Bz * rt6*( (-x**4 + 6*x**2*z**2 + y**4 - 6*y**2*z**2) + 1j*(-2*x**3*y - 2*x*y**3 + 12*x*y*z**2) )
        elif m==3:
            Bx = Bx * -( (4*x**4 - 21*x**2*y**2 - 3*x**2*z**2 + 3*y**4 + 3*y**2*z**2) + 1j*(15*x**3*y - 13*x*y**3 - 6*x*y*z**2) )
            By = By * -( (13*x**3*y - 15*x*y**3 + 6*x*y*z**2) + 1j*(-3*x**4 + 21*x**2*y**2 - 3*x**2*z**2 - 4*y**4 + 3*y**2*z**2) )
            Bz = Bz * -( (7*x**3*z - 21*x*y**2*z) + 1j*(21*x**2*y*z - 7*y**3*z) )
        else:
            print(" m = ", m)
            raise ValueError("In field_xyz.eval_Bi, n=3 and m is not between -n and n.")

    elif n==4:
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #   Hexadecapole moment components
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        A4 = 1/8*np.sqrt(2/np.pi)
        Bx = A4*Binm/r**11
        By = Bx + 0.0
        Bz = Bx + 0.0

        rt7o8 = sqrt(7/8)
        rt7 = sqrt(7)
        rt2 = sqrt(2)
        rt5o4 = sqrt(5/4)

        if m==-4:
            Bx = Bx * rt7o8*( (5*x**5 - 46*x**3*y**2 + 21*x*y**4 - 4*x**3*z**2 + 12*x*y**2*z**2) + 1j*(-24*x**4*y + 44*x**2*y**3 - 4*y**5 + 12*x**2*y*z**2 - 4*y**3*z**2) )
            By = By * rt7o8*( (21*x**4*y - 46*x**2*y**3 + 5*y**5 + 12*x**2*y*z**2 - 4*y**3*z**2) + 1j*(4*x**5 - 44*x**3*y**2 + 24*x*y**4 + 4*x**3*z**2 - 12*x*y**2*z**2) )
            Bz = Bz * rt7o8*( (9*x**4*z - 54*x**2*y**2*z + 9*y**4*z) + 1j*(-36*x**3*y*z + 36*x*y**3*z) )
        elif m==-3:
            Bx = Bx * rt7*( (6*x**4*z - 27*x**2*y**2*z + 3*y**4*z - 3*x**2*z**3 + 3*y**2*z**3) + 1j*(-21*x**3*y*z + 15*x*y**3*z + 6*x*y*z**3) )
            By = By * rt7*( (15*x**3*y*z - 21*x*y**3*z + 6*x*y*z**3) + 1j*(3*x**4*z - 27*x**2*y**2*z + 6*y**4*z + 3*x**2*z**3 - 3*y**2*z**3) )
            Bz = Bz * rt7*( (-x**5 + 2*x**3*y**2 + 3*x*y**4 + 8*x**3*z**2 - 24*x*y**2*z**2) + 1j*(3*x**4*y + 2*x**2*y**3 - y**5 - 24*x**2*y*z**2 + 8*y**3*z**2) )
        elif m==-2:
            Bx = Bx * -rt2*( (5*x**5 - 4*x**3*y**2 - 9*x*y**4 - 46*x**3*z**2 + 66*x*y**2*z**2 + 12*x*z**4) + 1j*(-12*x**4*y - 10*x**2*y**3 + 2*y**5 + 102*x**2*y*z**2 - 10*y**3*z**2 - 12*y*z**4) )
            By = By * -rt2*( (9*x**4*y + 4*x**2*y**3 - 5*y**5 - 66*x**2*y*z**2 + 46*y**3*z**2 - 12*y*z**4) + 1j*(2*x**5 - 10*x**3*y**2 - 12*x*y**4 - 10*x**3*z**2 + 102*x*y**2*z**2 - 12*x*z**4) )
            Bz = Bz * -rt2*( (21*x**4*z - 21*y**4*z - 42*x**2*z**3 + 42*y**2*z**3) + 1j*(-42*x**3*y*z - 42*x*y**3*z + 84*x*y*z**3) )
        elif m==-1:
            Bx = Bx * -( (18*x**4*z + 15*x**2*y**2*z - 3*y**4*z - 41*x**2*z**3 + y**2*z**3 + 4*z**5) + 1j*(-21*x**3*y*z - 21*x*y**3*z + 42*x*y*z**3) )
            By = By * -( (21*x**3*y*z + 21*x*y**3*z - 42*x*y*z**3) + 1j*(3*x**4*z - 15*x**2*y**2*z - 18*y**4*z - x**2*z**3 + 41*y**2*z**3 - 4*z**5) )
            Bz = Bz * -( (-3*x**5 - 6*x**3*y**2 - 3*x*y**4 + 36*x**3*z**2 + 36*x*y**2*z**2 - 24*x*z**4) + 1j*(3*x**4*y + 6*x**2*y**3 + 3*y**5 - 36*x**2*y*z**2 - 36*y**3*z**2 + 24*y*z**4) )
        elif m==0:
            Bx = Bx * rt5o4*(3*x**5 + 6*x**3*y**2 + 3*x*y**4 - 36*x**3*z**2 - 36*x*y**2*z**2 + 24*x*z**4)
            By = By * rt5o4*(3*x**4*y + 6*x**2*y**3 + 3*y**5 - 36*x**2*y*z**2 - 36*y**3*z**2 + 24*y*z**4)
            Bz = Bz * rt5o4*(15*x**4*z + 30*x**2*y**2*z + 15*y**4*z - 40*x**2*z**3 - 40*y**2*z**3 + 8*z**5)
        elif m==1:
            Bx = Bx * ( (18*x**4*z + 15*x**2*y**2*z - 3*y**4*z - 41*x**2*z**3 + y**2*z**3 + 4*z**5) + 1j*(21*x**3*y*z + 21*x*y**3*z - 42*x*y*z**3) )
            By = By * ( (21*x**3*y*z + 21*x*y**3*z - 42*x*y*z**3) + 1j*(-3*x**4*z + 15*x**2*y**2*z + 18*y**4*z + x**2*z**3 - 41*y**2*z**3 + 4*z**5) )
            Bz = Bz * ( (-3*x**5 - 6*x**3*y**2 - 3*x*y**4 + 36*x**3*z**2 + 36*x*y**2*z**2 - 24*x*z**4) + 1j*(-3*x**4*y - 6*x**2*y**3 - 3*y**5 + 36*x**2*y*z**2 + 36*y**3*z**2 - 24*y*z**4) )
        elif m==2:
            Bx = Bx * -rt2*( (5*x**5 - 4*x**3*y**2 - 9*x*y**4 - 46*x**3*z**2 + 66*x*y**2*z**2 + 12*x*z**4) + 1j*(12*x**4*y + 10*x**2*y**3 - 2*y**5 - 102*x**2*y*z**2 + 10*y**3*z**2 + 12*y*z**4) )
            By = By * -rt2*( (9*x**4*y + 4*x**2*y**3 - 5*y**5 - 66*x**2*y*z**2 + 46*y**3*z**2 - 12*y*z**4) + 1j*(-2*x**5 + 10*x**3*y**2 + 12*x*y**4 + 10*x**3*z**2 - 102*x*y**2*z**2 + 12*x*z**4) )
            Bz = Bz * -rt2*( (21*x**4*z - 21*y**4*z - 42*x**2*z**3 + 42*y**2*z**3) + 1j*(42*x**3*y*z + 42*x*y**3*z - 84*x*y*z**3) )
        elif m==3:
            Bx = Bx * -rt7*( (6*x**4*z - 27*x**2*y**2*z + 3*y**4*z - 3*x**2*z**3 + 3*y**2*z**3) + 1j*(21*x**3*y*z - 15*x*y**3*z - 6*x*y*z**3) )
            By = By * -rt7*( (15*x**3*y*z - 21*x*y**3*z + 6*x*y*z**3) + 1j*(-3*x**4*z + 27*x**2*y**2*z - 6*y**4*z - 3*x**2*z**3 + 3*y**2*z**3) )
            Bz = Bz * -rt7*( (-x**5 + 2*x**3*y**2 + 3*x*y**4 + 8*x**3*z**2 - 24*x*y**2*z**2) + 1j*(-3*x**4*y - 2*x**2*y**3 + y**5 + 24*x**2*y*z**2 - 8*y**3*z**2) )
        elif m==4:
            Bx = Bx * rt7o8*( (5*x**5 - 46*x**3*y**2 + 21*x*y**4 - 4*x**3*z**2 + 12*x*y**2*z**2) + 1j*(24*x**4*y - 44*x**2*y**3 + 4*y**5 - 12*x**2*y*z**2 + 4*y**3*z**2) )
            By = By * rt7o8*( (21*x**4*y - 46*x**2*y**3 + 5*y**5 + 12*x**2*y*z**2 - 4*y**3*z**2) + 1j*(-4*x**5 + 44*x**3*y**2 - 24*x*y**4 - 4*x**3*z**2 + 12*x*y**2*z**2) )
            Bz = Bz * rt7o8*( (9*x**4*z - 54*x**2*y**2*z + 9*y**4*z) + 1j*(36*x**3*y*z - 36*x*y**3*z) )
        else:
            print(" m = ", m)
            raise ValueError("In field_xyz.eval_Bi, n=4 and m is not between -n and n.")

    else:
        print(" n = ", n)
        raise ValueError("In field_xyz.eval_Bi, n>4 but only n=1 to n=4 are supported.")

    Bx = Bx * timeRot
    By = By * timeRot
    Bz = Bz * timeRot

    return Bx, By, Bz
    
#############################################

"""
eval_Bi_Schmidt()
    Same as eval_Bi but using Schmidt semi-normalized harmonics, with particular magnetic moments g_nm, h_nm.
    Usage: `Bx`, `By`, `Bz` = eval_Bi_Schmidt(`n`, `m`, `g_nm`, `h_nm`, `x`, `y`, `z`, `r`, `omega=None`, `t=None`)
    Returns:
        Bx, By, Bz (each): complex, ndarray shape(Nvals). A linear array of field values due to these particular n,m values,
            at each of the specified points. Returns a time sequence if ω and t are passed.
    Parameters:
        n: integer. Degree of magnetic moment to be evaluated.
        m: integer. Order of magnetic moment to be evaluated.
        g_nm, h_nm: complex. Magnetic moment of degree and order n,m. Units match the output field.
        x,y,z,r: float, shape(Nvals). Linear arrays of corresponding x,y, and z values. r is redundant but
            saves on computation time to avoid recalculating on every call to this function. If omega and t
            are passed, these quantities are the trajectory locations.
        omega: float (None). Optional oscillation frequency in rads/s for evaluating time series. Requires t to be passed as well.
        t: float, shape(Nvals) (None). Optional time values in TDB seconds since J2000 epoch. Required if omega is passed.
    """
def eval_Bi_Schmidt(n,m,g_nm,h_nm, x,y,z,r, omega=None, t=None):

    Bx, By, Bz = (np.zeros(np.size(r), dtype=np.complex_) for _ in range(3))

    if omega is None:
        timeRot = 1.0
    else:
        timeRot = np.exp(-1j*omega*t)

    if n==1:
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #   Dipole field components
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        A1 = 1.0
        Bx = A1/r**5
        By = Bx + 0.0
        Bz = Bx + 0.0

        if m==0:
            Bx = Bx * g_nm * (3*x*z)
            By = By * g_nm * (3*y*z)
            Bz = Bz * g_nm * (-x**2 - y**2 + 2*z**2)
        elif m==1:
            Bx = Bx * ( g_nm*(2*x**2 - y**2 - z**2) + h_nm*(3*x*y) )
            By = By * ( g_nm*(3*x*y) + h_nm*(-x**2 + 2*y**2 - z**2) )
            Bz = Bz * ( g_nm*(3*x*z) + h_nm*(3*y*z) )
        else:
            print(" m = ", m)
            raise ValueError("In field_xyz.eval_Bi_Schmidt, n=1 and m is not between 0 and n.")

    elif n==2:
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #   Quadrupole moment components
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        rt3 = sqrt(3)

        A2 = 1/2*rt3
        Bx = A2/r**7
        By = Bx + 0.0
        Bz = Bx + 0.0

        if m==0:
            Bx = Bx * g_nm * -rt3*(x**3 + x*y**2 - 4*x*z**2)
            By = By * g_nm * -rt3*(x**2*y + y**3 - 4*y*z**2)
            Bz = Bz * g_nm * -rt3*(3*x**2*z + 3*y**2*z - 2*z**3)
        elif m==1:
            Bx = Bx * -2*( g_nm*(-4*x**2*z + y**2*z + z**3) + h_nm*(-5*x*y*z) )
            By = By * -2*( g_nm*(-5*x*y*z) + h_nm*(x**2*z - 4*y**2*z + z**3) )
            Bz = Bz * -2*( g_nm*(x**3 + x*y**2 - 4*x*z**2) + h_nm*(x**2*y + y**3 - 4*y*z**2) )
        elif m==2:
            Bx = Bx * ( g_nm*(3*x**3 - 7*x*y**2 - 2*x*z**2) + h_nm*(8*x**2*y - 2*y**3 - 2*y*z**2) )
            By = By * ( g_nm*(7*x**2*y - 3*y**3 + 2*y*z**2) + h_nm*(-2*x**3 + 8*x*y**2 - 2*x*z**2) )
            Bz = Bz * ( g_nm*(5*x**2*z - 5*y**2*z) + h_nm*(10*x*y*z) )
        else:
            print(" m = ", m)
            raise ValueError("In field_xyz.eval_Bi_Schmidt, n=2 and m is not between 0 and n.")

    elif n==3:
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #   Octupole moment components
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        rt2o3 = sqrt(2/3)
        rt6 = sqrt(6)
        rt10 = sqrt(10)
        rt15 = sqrt(15)

        A3 = 1/2/rt2o3
        Bx = A3/r**9
        By = Bx + 0.0
        Bz = Bx + 0.0

        if m==0:
            Bx = Bx * g_nm * 5*rt6*(-x**3*z - x*y**2*z + 4/3*x*z**3)
            By = By * g_nm * 5*rt6*(-x**2*y*z - y**3*z + 4/3*y*z**3)
            Bz = Bz * g_nm * rt6*(x**4 + 2*x**2*y**2 - 8*x**2*z**2 + y**4 - 8*y**2*z**2 + 8/3*z**4)
        elif m==1:
            Bx = Bx * ( g_nm*(-4*x**4 - 3*x**2*y**2 + 27*x**2*z**2 + y**4 - 3*y**2*z**2 - 4*z**4) + h_nm*(-5*x**3*y - 5*x*y**3 + 30*x*y*z**2) )
            By = By * ( g_nm*(-5*x**3*y - 5*x*y**3 + 30*x*y*z**2) + h_nm*(x**4 - 3*x**2*y**2 - 3*x**2*z**2 - 4*y**4 + 27*y**2*z**2 - 4*z**4) )
            Bz = Bz * ( g_nm*(-15*x**3*z - 15*x*y**2*z + 20*x*z**3) + h_nm*(-15*x**2*y*z - 15*y**3*z + 20*y*z**3) )
        elif m==2:
            Bx = Bx * rt10*( g_nm*(5*x**3*z - 9*x*y**2*z - 2*x*z**3) + h_nm*(12*x**2*y*z - 2*y**3*z - 2*y*z**3) )
            By = By * rt10*( g_nm*(9*x**2*y*z - 5*y**3*z + 2*y*z**3) + h_nm*(-2*x**3*z + 12*x*y**2*z - 2*x*z**3) )
            Bz = Bz * rt10*( g_nm*(-x**4 + 6*x**2*z**2 + y**4 - 6*y**2*z**2) + h_nm*(-2*x**3*y - 2*x*y**3 + 12*x*y*z**2) )
        elif m==3:
            Bx = Bx * rt15*( g_nm*(4/3*x**4 - 7*x**2*y**2 - x**2*z**2 + y**4 + y**2*z**2) + h_nm*(5*x**3*y - 13/3*x*y**3 - 2*x*y*z**2) )
            By = By * rt15*( g_nm*(13/3*x**3*y - 5*x*y**3 + 2*x*y*z**2) + h_nm*(-x**4 + 7*x**2*y**2 - x**2*z**2 - 4/3*y**4 + y**2*z**2) )
            Bz = Bz * rt15*( g_nm*(7/3*x**3*z - 7*x*y**2*z) + h_nm*(7*x**2*y*z - 7/3*y**3*z) )
        else:
            raise ValueError("In field_xyz.eval_Bi_Schmidt, n=3 and m is not between 0 and n.")

    elif n==4:
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #   Hexadecapole moment components
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        rt2 = sqrt(2)
        rt5 = sqrt(5)
        rt7 = sqrt(7)
        rt7o2 = rt7/rt2

        A4 = 1/2*rt5
        Bx = A4/r**11
        By = Bx + 0.0
        Bz = Bx + 0.0

        if m==0:
            Bx = Bx * g_nm * 3/4*rt5*(x**5 + 2*x**3*y**2 + x*y**4 - 12*x**3*z**2 - 12*x*y**2*z**2 + 8*x*z**4)
            By = By * g_nm * 3/4*rt5*(x**4*y + 2*x**2*y**3 + y**5 - 12*x**2*y*z**2 - 12*y**3*z**2 + 8*y*z**4)
            Bz = Bz * g_nm * 3/4*rt5*(5*x**4*z + 10*x**2*y**2*z + 5*y**4*z - 40/3*x**2*z**3 - 40/3*y**2*z**3 + 8/3*z**5)
        elif m==1:
            Bx = Bx / rt2*( g_nm*(-18*x**4*z - 15*x**2*y**2*z + 3*y**4*z + 41*x**2*z**3 - y**2*z**3 - 4*z**5) + h_nm*(-21*x**3*y*z - 21*x*y**3*z + 42*x*y*z**3) )
            By = By / rt2*( g_nm*(-21*x**3*y*z - 21*x*y**3*z + 42*x*y*z**3) + h_nm*(3*x**4*z - 15*x**2*y**2*z - 18*y**4*z - x**2*z**3 + 41*y**2*z**3 - 4*z**5) )
            Bz = Bz / rt2*( g_nm*(3*x**5 + 6*x**3*y**2 + 3*x*y**4 - 36*x**3*z**2 - 36*x*y**2*z**2 + 24*x*z**4) + h_nm*(3*x**4*y + 6*x**2*y**3 + 3*y**5 - 36*x**2*y*z**2 - 36*y**3*z**2 + 24*y*z**4) )
        elif m==2:
            Bx = Bx * ( g_nm*(-5/2*x**5 + 2*x**3*y**2 + 9/2*x*y**4 + 23*x**3*z**2 - 33*x*y**2*z**2 - 6*x*z**4) + h_nm*(-6*x**4*y - 5*x**2*y**3 + y**5 + 51*x**2*y*z**2 - 5*y**3*z**2 - 6*y*z**4) )
            By = By * ( g_nm*(-9/2*x**4*y - 2*x**2*y**3 + 5/2*y**5 + 33*x**2*y*z**2 - 23*y**3*z**2 + 6*y*z**4) + h_nm*(x**5 - 5*x**3*y**2 - 6*x*y**4 - 5*x**3*z**2 + 51*x*y**2*z**2 - 6*x*z**4) )
            Bz = Bz * ( g_nm*(-21/2*x**4*z + 21/2*y**4*z + 21*x**2*z**3 - 21*y**2*z**3) + h_nm*(-21*x**3*y*z - 21*x*y**3*z + 42*x*y*z**3) )
        elif m==3:
            Bx = Bx * rt7o2*( g_nm*(6*x**4*z - 27*x**2*y**2*z + 3*y**4*z - 3*x**2*z**3 + 3*y**2*z**3) + h_nm*(21*x**3*y*z - 15*x*y**3*z - 6*x*y*z**3) )
            By = By * rt7o2*( g_nm*(15*x**3*y*z - 21*x*y**3*z + 6*x*y*z**3) + h_nm*(-3*x**4*z + 27*x**2*y**2*z - 6*y**4*z - 3*x**2*z**3 + 3*y**2*z**3) )
            Bz = Bz * rt7o2*( g_nm*(-x**5 + 2*x**3*y**2 + 3*x*y**4 + 8*x**3*z**2 - 24*x*y**2*z**2) + h_nm*(-3*x**4*y - 2*x**2*y**3 + y**5 + 24*x**2*y*z**2 - 8*y**3*z**2) )
        elif m==4:
            Bx = Bx * rt7*( g_nm*(5/4*x**5 - 23/2*x**3*y**2 + 21/4*x*y**4 - x**3*z**2 + 3*x*y**2*z**2) + h_nm*(6*x**4*y - 11*x**2*y**3 + y**5 - 3*x**2*y*z**2 + y**3*z**2) )
            By = By * rt7*( g_nm*(21/4*x**4*y - 23/2*x**2*y**3 + 5/4*y**5 + 3*x**2*y*z**2 - y**3*z**2) + h_nm*(-x**5 + 11*x**3*y**2 - 6*x*y**4 - x**3*z**2 + 3*x*y**2*z**2) )
            Bz = Bz * rt7*( g_nm*(9/4*x**4*z - 27/2*x**2*y**2*z + 9/4*y**4*z) + h_nm*(9*x**3*y*z - 9*x*y**3*z) )
        else:
            print(" m = ", m)
            raise ValueError("In field_xyz.eval_Bi_Schmidt, n=4 and m is not between 0 and n.")

    else:
        print(" n = ", n)
        raise ValueError("In field_xyz.eval_Bi_Schmidt, n>4 but only n=1 to n=4 are supported.")

    Bx = Bx * timeRot
    By = By * timeRot
    Bz = Bz * timeRot

    return Bx, By, Bz
