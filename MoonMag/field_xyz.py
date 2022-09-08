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

        A1 = sqrt(3/8/np.pi)
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
            raise ValueError(f"In field_xyz.eval_Bi, n={n} and m is not between -n and n.")

    elif n==2:
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #   Linear field components
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        A2 = sqrt(15/32/np.pi)
        B_base = 2*A2*Benm

        if m==-2:
            Bx = B_base * (-x + 1j*y)
            By = B_base * (y + 1j*x)
            Bz = 0
        elif m==-1:
            Bx = B_base * -z
            By = B_base * 1j*z
            Bz = B_base * (-x + 1j*y)
        elif m==0:
            rt2o3 = sqrt(2/3)
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
            raise ValueError(f"In field_xyz.eval_Bi, n={n} and m is not between -n and n.")

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

        A1 = sqrt(3/8/np.pi)
        Bx = A1*Binm/r**5
        By = Bx + 0.0
        Bz = Bx + 0.0

        if m==-1:
            Bx = Bx * ( (2*x**2 - y**2 - z**2) + 1j*(-3*x*y) )
            By = By * ( (3*x*y) + 1j*(x**2 - 2*y**2 + z**2) )
            Bz = Bz * ( (3*x*z) + 1j*(-3*y*z) )
        elif m==0:
            rt2 = sqrt(2)
            Bx = Bx * 3*rt2*(x*z)
            By = By * 3*rt2*(y*z)
            Bz = Bz * -rt2*(x**2 + y**2 - 2*z**2)
        elif m==1:
            Bx = Bx * ( (-2*x**2 + y**2 + z**2) + 1j*(-3*x*y) )
            By = By * ( (-3*x*y) + 1j*(x**2 - 2*y**2 + z**2) )
            Bz = Bz * ( (-3*x*z) + 1j*(-3*y*z) )
        else:
            print(" m = ", m)
            raise ValueError(f"In field_xyz.eval_Bi, n={n} and m is not between -n and n.")

    elif n==2:
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #   Quadrupole moment components
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        A2 = sqrt(15/32/np.pi)
        Bx = A2*Binm/r**7
        By = Bx + 0.0
        Bz = Bx + 0.0

        if m==-2:
            Bx = Bx * ( x*(3*x**2 - 7*y**2 - 2*z**2) + 1j*2*y*(-4*x**2 + y**2 + z**2) )
            By = By * ( y*(7*x**2 - 3*y**2 + 2*z**2) + 1j*2*x*(x**2 - 4*y**2 + 2*z**2) )
            Bz = Bz * ( 5*(x**2 - y**2)*z + 1j*(-10*x*y*z) )
        elif m==-1:
            Bx = Bx * ( (8*x**2*z - 2*z*(y**2 + z**2)) + 1j*(-10*x*y*z) )
            By = By * ( (10*x*y*z) + 1j*2*z*(x**2 - 4*y**2 + z**2) )
            Bz = Bz * ( -2*x*(x**2 + y**2 - 4*z**2) + 1j*2*y*(x**2 + y**2 - 4*z**2) )
        elif m==0:
            rt6 = sqrt(6)
            Bx = Bx * -rt6*x*(x**2 + y**2 - 4*z**2)
            By = By * -rt6*y*(x**2 + y**2 - 4*z**2)
            Bz = Bz * rt6*z*(-3*x**2 - 3*y**2 + 2*z**2)
        elif m==1:
            Bx = Bx * ( 2*z*(-4*x**2 + y**2 + z**2) + 1j*(-10*x*y*z) )
            By = By * ( (-10*x*y*z) + 1j*2*z*(x**2 - 4*y**2 + z**2) )
            Bz = Bz * ( 2*x*(x**2 + y**2 - 4*z**2) + 1j*2*y*(x**2 + y**2 - 4*z**2) )
        elif m==2:
            Bx = Bx * ( x*(3*x**2 - 7*y**2 - 2*z**2) + 1j*(8*x**2*y - 2*y*(y**2 + z**2)) )
            By = By * ( y*(7*x**2 - 3*y**2 + 2*z**2) + 1j*-2*x*(x**2 - 4*y**2 + 2*z**2) )
            Bz = Bz * ( 5*(x**2 - y**2)*z + 1j*(10*x*y*z) )
        else:
            print(" m = ", m)
            raise ValueError(f"In field_xyz.eval_Bi, n={n} and m is not between -n and n.")

    elif n==3:
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #   Octupole moment components
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        A3 = sqrt(7/64/np.pi)
        Bx = A3*Binm/r**9
        By = Bx + 0.0
        Bz = Bx + 0.0

        if m==-3:
            rt5 = sqrt(5)
            Bx = Bx * rt5*( (4*x**4 + 3*y**2*(y**2 + z**2) - 3*x**2*(7*y**2 + z**2)) + 1j*x*y*(-15*x**2 + 13*y**2 + 6*z**2) )
            By = By * rt5*( x*y*(13*x**2 - 15*y**2 + 6*z**2) + 1j*(3*x**4 + 4*y**4 - 3*y**2*z**2 + 3*x**2*(-7*y**2 + z**2)) )
            Bz = Bz * 7*rt5*( x*(x**2 - 3*y**2)*z + 1j*y*(-3*x**2 + y**2)*z )
        elif m==-2:
            rt30 = sqrt(30)
            Bx = Bx * rt30*( x*z*(5*x**2 - 9*y**2 - 2*z**2) + 1j*2*y*z*(-6*x**2 + y**2 + z**2) )
            By = By * rt30*( y*z*(9*x**2 - 5*y**2 + 2*z**2) + 1j*2*x*z*(x**2 - 6*y**2 + z**2) )
            Bz = Bz * rt30*( -(x**2 - y**2)*(x**2 + y**2 - 6*z**2) + 1j*2*x*y*(x**2 + y**2 - 6*z**2) )
        elif m==-1:
            rt3 = sqrt(3)
            Bx = Bx * rt3*( (-4*x**4 + y**4 - 3*y**2*z**2 - 4*z**4 - 3*x**2*(y**2 - 9*z**2)) + 1j*5*x*y*(x**2 + y**2 - 6*z**2) )
            By = By * rt3*( -5*x*y*(x**2 + y**2 - 6*z**2) + 1j*(-x**4 + 4*y**4 - 27*y**2*z**2 + 4*z**4 + 3*x**2*(y**2 + z**2)) )
            Bz = Bz * rt3*( -5*x*z*(3*x**2 + 3*y**2 - 4*z**2) + 1j*5*y*z*(3*x**2 + 3*y**2 - 4*z**2) )
        elif m==0:
            Bx = Bx * -10*x*z*(3*x**2 + 3*y**2 - 4*z**2)
            By = By * -10*y*z*(3*x**2 + 3*y**2 - 4*z**2)
            Bz = Bz * 2*(3*x**4 + 3*y**4 - 24*y**2*z**2 + 8*z**4 + 6*x**2*(y**2 - 4*z**2))
        elif m==1:
            rt3 = sqrt(3)
            Bx = Bx * rt3*( (4*x**4 - y**4 + 3*y**2*z**2 + 4*z**4 + 3*x**2*(y**2 - 9*z**2)) + 1j*5*x*y*(x**2 + y**2 - 6*z**2) )
            By = By * rt3*( 5*x*y*(x**2 + y**2 - 6*z**2) + 1j*(-x**4 + 4*y**4 - 27*y**2*z**2 + 4*z**4 + 3*x**2*(y**2 + z**2)) )
            Bz = Bz * 5*rt3*( x*z*(3*x**2 + 3*y**2 - 4*z**2) + 1j*y*z*(3*x**2 + 3*y**2 - 4*z**2) )
        elif m==2:
            rt30 = sqrt(30)
            Bx = Bx * rt30*( x*z*(5*x**2 - 9*y**2 - 2*z**2) + 1j*-2*y*z*(-6*x**2 + y**2 + z**2) )
            By = By * rt30*( y*z*(9*x**2 - 5*y**2 + 2*z**2) + 1j*-2*x*z*(x**2 - 6*y**2 + z**2) )
            Bz = Bz * rt30*( -(x**2 - y**2)*(x**2 + y**2 - 6*z**2) + 1j*-2*x*y*(x**2 + y**2 - 6*z**2) )
        elif m==3:
            rt5 = sqrt(5)
            Bx = Bx * rt5*( (-4*x**4 - 3*y**2*(y**2 + z**2) + 3*x**2*(7*y**2 + z**2)) + 1j*x*y*(-15*x**2 + 13*y**2 + 6*z**2) )
            By = By * rt5*( x*y*(-13*x**2 + 15*y**2 - 6*z**2) + 1j*(3*x**4 + 4*y**4 - 3*y**2*z**2 + 3*x**2*(-7*y**2 + z**2)) )
            Bz = Bz * 7*rt5*( -x*(x**2 - 3*y**2)*z + 1j*y*(-3*x**2 + y**2)*z )
        else:
            print(" m = ", m)
            raise ValueError(f"In field_xyz.eval_Bi, n={n} and m is not between -n and n.")

    elif n==4:
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #   Hexadecapole moment components
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        A4 = sqrt(1/192/np.pi)
        Bx = A4*Binm/r**11
        By = Bx + 0.0
        Bz = Bx + 0.0

        if m==-4:
            rt2 = sqrt(2)
            rt105 = sqrt(105)
            Bx = Bx * 3*rt105*( 1/2/rt2*x*(5*x**4 - 2*x**2*(23*y**2 + 2*z**2) + 3*y**2*(7*y**2 + 4*z**2)) + 1j*-rt2*y*(6*x**4 + y**2*(y**2 + z**2) - x**2*(11*y**2 + 3*z**2)) )
            By = By * 3*rt105*( 1/2/rt2*y*(21*x**4 + 5*y**4 - 4*y**2*z**2 + x**2*(-46*y**2 + 12*z**2)) + 1j*rt2*x*(x**4 + 6*y**4 - 3*y**2*z**2 + x**2*(-11*y**2 + z**2)) )
            Bz = Bz * 27*rt105*( 1/2/rt2*(x**4 - 6*x**2*y**2 + y**4)*z + 1j*rt2*x*y*(-x**2 + y**2)*z )
        elif m==-3:
            rt105 = sqrt(105)
            Bx = Bx * 9*rt105*( z*(2*x**4 + y**2*(y**2 + z**2) - x**2*(9*y**2 + z**2)) + 1j*x*y*z*(-7*x**2 + 5*y**2 + 2*z**2) )
            By = By * 9*rt105*( x*y*z*(5*x**2 - 7*y**2 + 2*z**2) + 1j*z*(x**4 + 2*y**4 - y**2*z**2 + x**2*(-9*y**2 + z**2)) )
            Bz = Bz * -3*rt105*( x*(x**2 - 3*y**2)*(x**2 + y**2 - 8*z**2) + 1j*y*(-3*x**2 + y**2)*(x**2 + y**2 - 8*z**2) )
        elif m==-2:
            rt2 = sqrt(2)
            rt15 = sqrt(15)
            Bx = Bx * 3*rt15*( -1/rt2*x*(5*x**4 - 9*y**4 + 66*y**2*z**2 + 12*z**4 - 2*x**2*(2*y**2 + 23*z**2)) + 1j*rt2*y*(6*x**4 - y**4 + 5*y**2*z**2 + 6*z**4 + x**2*(5*y**2 - 51*z**2)) )
            By = By * 3*rt15*( 1/rt2*y*(-9*x**4 + 5*y**4 - 46*y**2*z**2 + 12*z**4 + x**2*(-4*y**2 + 66*z**2)) + 1j*-rt2*x*(x**4 - 6*y**4 + 51*y**2*z**2 - 6*z**4 - 5*x**2*(y**2 + z**2)) )
            Bz = Bz * 63*rt15*( -1/rt2*(x**2 - y**2)*z*(x**2 + y**2 - 2*z**2) + 1j*rt2*x*y*z*(x**2 + y**2 - 2*z**2) )
        elif m==-1:
            rt15 = sqrt(15)
            Bx = Bx * 3*rt15*( -z*(18*x**4 - 3*y**4 + y**2*z**2 + 4*z**4 + x**2*(15*y**2 - 41*z**2)) + 1j*21*x*y*z*(x**2 + y**2 - 2*z**2) )
            By = By * 3*rt15*( -21*x*y*z*(x**2 + y**2 - 2*z**2) + 1j*z*(-3*x**4 + 18*y**4 - 41*y**2*z**2 + 4*z**4 + x**2*(15*y**2 + z**2)) )
            Bz = Bz * 9*rt15*( x*(x**4 + y**4 - 12*y**2*z**2 + 8*z**4 + 2*x**2*(y**2 - 6*z**2)) + 1j*-y*(x**4 + y**4 - 12*y**2*z**2 + 8*z**4 + 2*x**2*(y**2 - 6*z**2)) )
        elif m==0:
            rt3 = sqrt(3)
            Bx = Bx * 45/2*rt3*x*(x**4 + y**4 - 12*y**2*z**2 + 8*z**4 + 2*x**2*(y**2 - 6*z**2))
            By = By * 45/2*rt3*y*(x**4 + y**4 - 12*y**2*z**2 + 8*z**4 + 2*x**2*(y**2 - 6*z**2))
            Bz = Bz * 15/2*rt3*z*(15*x**4 + 15*y**4 - 40*y**2*z**2 + 8*z**4 + 10*x**2*(3*y**2 - 4*z**2))
        elif m==1:
            rt15 = sqrt(15)
            Bx = Bx * 3*rt15*( z*(18*x**4 - 3*y**4 + y**2*z**2 + 4*z**4 + x**2*(15*y**2 - 41*z**2)) + 1j*21*x*y*z*(x**2 + y**2 - 2*z**2) )
            By = By * 3*rt15*( 21*x*y*z*(x**2 + y**2 - 2*z**2) + 1j*z*(-3*x**4 + 18*y**4 - 41*y**2*z**2 + 4*z**4 + x**2*(15*y**2 + z**2)) )
            Bz = Bz * -9*rt15*( x*(x**4 + y**4 - 12*y**2*z**2 + 8*z**4 + 2*x**2*(y**2 - 6*z**2)) + 1j*y*(x**4 + y**4 - 12*y**2*z**2 + 8*z**4 + 2*x**2*(y**2 - 6*z**2)) )
        elif m==2:
            rt2 = sqrt(2)
            rt15 = sqrt(15)
            Bx = Bx * 3*rt15*( -1/rt2*x*(5*x**4 - 9*y**4 + 66*y**2*z**2 + 12*z**4 - 2*x**2*(2*y**2 + 23*z**2)) + 1j*rt2*y*(-6*x**4 + y**4 - 5*y**2*z**2 - 6*z**4 + x**2*(-5*y**2 + 51*z**2)) )
            By = By * 3*rt15*( 1/rt2*y*(-9*x**4 + 5*y**4 - 46*y**2*z**2 + 12*z**4 + x**2*(-4*y**2 + 66*z**2)) + 1j*rt2*x*(x**4 - 6*y**4 + 51*y**2*z**2 - 6*z**4 - 5*x**2*(y**2 + z**2)) )
            Bz = Bz * -63*rt15*( 1/rt2*(x**2 - y**2)*z*(x**2 + y**2 - 2*z**2) + 1j*rt2*x*y*z*(x**2 + y**2 - 2*z**2) )
        elif m==3:
            rt105 = sqrt(105)
            Bx = Bx * 9*rt105*( -z*(2*x**4 + y**2*(y**2 + z**2) - x**2*(9*y**2 + z**2)) + 1j*x*y*z*(-7*x**2 + 5*y**2 + 2*z**2) )
            By = By * 9*rt105*( -x*y*z*(5*x**2 - 7*y**2 + 2*z**2) + 1j*z*(x**4 + 2*y**4 - y**2*z**2 + x**2*(-9*y**2 + z**2)) )
            Bz = Bz * 3*rt105*( x*(x**2 - 3*y**2)*(x**2 + y**2 - 8*z**2) + 1j*-y*(-3*x**2 + y**2)*(x**2 + y**2 - 8*z**2) )
        elif m==4:
            rt2 = sqrt(2)
            rt105 = sqrt(105)
            Bx = Bx * 3*rt105*( 1/2/rt2*x*(5*x**4 - 2*x**2*(23*y**2 + 2*z**2) + 3*y**2*(7*y**2 + 4*z**2)) + 1j*rt2*y*(6*x**4 + y**2*(y**2 + z**2) - x**2*(11*y**2 + 3*z**2)) )
            By = By * 3*rt105*( 1/2/rt2*y*(21*x**4 + 5*y**4 - 4*y**2*z**2 + x**2*(-46*y**2 + 12*z**2)) + 1j*-rt2*x*(x**4 + 6*y**4 - 3*y**2*z**2 + x**2*(-11*y**2 + z**2)) )
            Bz = Bz * 27*rt105*( 1/2/rt2*(x**4 - 6*x**2*y**2 + y**4)*z + 1j*rt2*x*y*(x**2 - y**2)*z )
        else:
            print(" m = ", m)
            raise ValueError(f"In field_xyz.eval_Bi, n={n} and m is not between -n and n.")

    elif n==5:
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #   n=5 moment components
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        A5 = sqrt(11/16/np.pi)
        Bx = A5*Binm/r**13
        By = Bx + 0.0
        Bz = Bx + 0.0

        if m==-5:
            rt7 = sqrt(7)
            Bx = Bx * 3/8*rt7*( (6*x**6 - 5*y**4*(y**2 + z**2) - 5*x**4*(17*y**2 + z**2) + 10*x**2*(8*y**4 + 3*y**2*z**2)) + 1j*-x*y*(35*x**4 + 31*y**4 + 20*y**2*z**2 - 10*x**2*(11*y**2 + 2*z**2)) )
            By = By * 3/8*rt7*( x*y*(31*x**4 + 5*y**2*(7*y**2 - 4*z**2) + x**2*(-110*y**2 + 20*z**2)) + 1j*(5*x**6 - 6*y**6 + 5*y**4*z**2 + x**4*(-80*y**2 + 5*z**2) + 5*x**2*(17*y**4 - 6*y**2*z**2)) )
            Bz = Bz * 33/8*rt7*( x*(x**4 - 10*x**2*y**2 + 5*y**4)*z + 1j*-y*(5*x**4 - 10*x**2*y**2 + y**4)*z )
        elif m==-4:
            rt35o2 = sqrt(35/2)
            Bx = Bx * 3*rt35o2*( 1/4*x*z*(7*x**4 + 23*y**4 + 12*y**2*z**2 - 2*x**2*(29*y**2 + 2*z**2)) + 1j*-y*z*(8*x**4 + y**2*(y**2 + z**2) - x**2*(13*y**2 + 3*z**2)) )
            By = By * 3*rt35o2*( 1/4*y*z*(23*x**4 + 7*y**4 - 4*y**2*z**2 + x**2*(-58*y**2 + 12*z**2)) + 1j*x*z*(x**4 + 8*y**4 - 3*y**2*z**2 + x**2*(-13*y**2 + z**2)) )
            Bz = Bz * 3*rt35o2*( -1/4*(x**4 - 6*x**2*y**2 + y**4)*(x**2 + y**2 - 10*z**2) + 1j*x*y*(x**2 - y**2)*(x**2 + y**2 - 10*z**2) )
        elif m==-3:
            rt35 = sqrt(35)
            Bx = Bx * 3/8*rt35*( -(2*x**6 + y**6 - 7*y**4*z**2 - 8*y**2*z**4 - x**4*(7*y**2 + 23*z**2) + x**2*(-8*y**4 + 90*y**2*z**2 + 8*z**4)) + 1j*x*y*(7*x**4 - 5*y**4 + 44*y**2*z**2 + 16*z**4 + 2*x**2*(y**2 - 38*z**2)) )
            By = By * -3/8*rt35*( x*y*(5*x**4 - 7*y**4 + 76*y**2*z**2 - 16*z**4 - 2*x**2*(y**2 + 22*z**2)) + 1j*(x**6 + 2*y**6 - 23*y**4*z**2 + 8*y**2*z**4 - x**4*(8*y**2 + 7*z**2) + x**2*(-7*y**4 + 90*y**2*z**2 - 8*z**4)) )
            Bz = Bz * -9/8*rt35*( x*(x**2 - 3*y**2)*z*(3*x**2 + 3*y**2 - 8*z**2) + 1j*y*(-3*x**2 + y**2)*z*(3*x**2 + 3*y**2 - 8*z**2) )
        elif m==-2:
            rt105o2 = sqrt(105/2)
            Bx = Bx * rt105o2*( 1/2*x*z*(-7*x**4 + 11*y**4 - 26*y**2*z**2 - 4*z**4 + x**2*(4*y**2 + 22*z**2)) + 1j*y*z*(8*x**4 - y**4 + y**2*z**2 + 2*z**4 + x**2*(7*y**2 - 23*z**2)) )
            By = By * rt105o2*( 1/2*y*z*(-11*x**4 + 7*y**4 - 22*y**2*z**2 + 4*z**4 + x**2*(-4*y**2 + 26*z**2)) + 1j*x*z*(-x**4 + 8*y**4 - 23*y**2*z**2 + 2*z**4 + x**2*(7*y**2 + z**2)) )
            Bz = Bz * rt105o2*( 1/2*(x**2 - y**2)*(x**4 + y**4 - 16*y**2*z**2 + 16*z**4 + 2*x**2*(y**2 - 8*z**2)) + 1j*-x*y*(x**4 + y**4 - 16*y**2*z**2 + 16*z**4 + 2*x**2*(y**2 - 8*z**2)) )
        elif m==-1:
            rt15o2 = sqrt(15/2)
            Bx = Bx * 1/4*rt15o2*( (6*x**6 - y**6 + 11*y**4*z**2 + 4*y**2*z**4 - 8*z**6 + x**4*(11*y**2 - 101*z**2) + 2*x**2*(2*y**4 - 45*y**2*z**2 + 58*z**4)) + 1j*-7*x*y*(x**4 + y**4 - 16*y**2*z**2 + 16*z**4 + 2*x**2*(y**2 - 8*z**2)) )
            By = By * 1/4*rt15o2*( 7*x*y*(x**4 + y**4 - 16*y**2*z**2 + 16*z**4 + 2*x**2*(y**2 - 8*z**2)) + 1j*(x**6 - 6*y**6 + 101*y**4*z**2 - 116*y**2*z**4 + 8*z**6 - x**4*(4*y**2 + 11*z**2) + x**2*(-11*y**4 + 90*y**2*z**2 - 4*z**4)) )
            Bz = Bz * 7/4*rt15o2*( x*z*(5*x**4 + 5*y**4 - 20*y**2*z**2 + 8*z**4 + 10*x**2*(y**2 - 2*z**2)) + 1j*-y*z*(5*x**4 + 5*y**4 - 20*y**2*z**2 + 8*z**4 + 10*x**2*(y**2 - 2*z**2)) )
        elif m==0:
            Bx = Bx * 21/4*x*z*(5*x**4 + 5*y**4 - 20*y**2*z**2 + 8*z**4 + 10*x**2*(y**2 - 2*z**2))
            By = By * 21/4*y*z*(5*x**4 + 5*y**4 - 20*y**2*z**2 + 8*z**4 + 10*x**2*(y**2 - 2*z**2))
            Bz = Bz * -3/4*(5*x**6 + 5*y**6 - 90*y**4*z**2 + 120*y**2*z**4 - 16*z**6 + 15*x**4*(y**2 - 6*z**2) + 15*x**2*(y**4 - 12*y**2*z**2 + 8*z**4))
        elif m==1:
            rt15o2 = sqrt(15/2)
            Bx = Bx * 1/4*rt15o2*( (-6*x**6 + y**6 - 11*y**4*z**2 - 4*y**2*z**4 + 8*z**6 + x**4*(-11*y**2 + 101*z**2) - 2*x**2*(2*y**4 - 45*y**2*z**2 + 58*z**4)) + 1j*-7*x*y*(x**4 + y**4 - 16*y**2*z**2 + 16*z**4 + 2*x**2*(y**2 - 8*z**2)) )
            By = By * 1/4*rt15o2*( -7*x*y*(x**4 + y**4 - 16*y**2*z**2 + 16*z**4 + 2*x**2*(y**2 - 8*z**2)) + 1j*(x**6 - 6*y**6 + 101*y**4*z**2 - 116*y**2*z**4 + 8*z**6 - x**4*(4*y**2 + 11*z**2) + x**2*(-11*y**4 + 90*y**2*z**2 - 4*z**4)) )
            Bz = Bz * -7/4*rt15o2*( x*z*(5*x**4 + 5*y**4 - 20*y**2*z**2 + 8*z**4 + 10*x**2*(y**2 - 2*z**2)) + 1j*y*z*(5*x**4 + 5*y**4 - 20*y**2*z**2 + 8*z**4 + 10*x**2*(y**2 - 2*z**2)) )
        elif m==2:
            rt105o2 = sqrt(105/2)
            Bx = Bx * rt105o2*( 1/2*x*z*(-7*x**4 + 11*y**4 - 26*y**2*z**2 - 4*z**4 + x**2*(4*y**2 + 22*z**2)) + 1j*y*z*(-8*x**4 + y**4 - y**2*z**2 - 2*z**4 + x**2*(-7*y**2 + 23*z**2)) )
            By = By * rt105o2*( 1/2*y*z*(-11*x**4 + 7*y**4 - 22*y**2*z**2 + 4*z**4 + x**2*(-4*y**2 + 26*z**2)) + 1j*x*z*(x**4 - 8*y**4 + 23*y**2*z**2 - 2*z**4 - x**2*(7*y**2 + z**2)) )
            Bz = Bz * rt105o2*( 1/2*(x**2 - y**2)*(x**4 + y**4 - 16*y**2*z**2 + 16*z**4 + 2*x**2*(y**2 - 8*z**2)) + 1j*x*y*(x**4 + y**4 - 16*y**2*z**2 + 16*z**4 + 2*x**2*(y**2 - 8*z**2)) )
        elif m==3:
            rt35 = sqrt(35)
            Bx = Bx * 3/8*rt35*( (2*x**6 + y**6 - 7*y**4*z**2 - 8*y**2*z**4 - x**4*(7*y**2 + 23*z**2) + x**2*(-8*y**4 + 90*y**2*z**2 + 8*z**4)) + 1j*x*y*(7*x**4 - 5*y**4 + 44*y**2*z**2 + 16*z**4 + 2*x**2*(y**2 - 38*z**2)) )
            By = By * 3/8*rt35*( x*y*(5*x**4 - 7*y**4 + 76*y**2*z**2 - 16*z**4 - 2*x**2*(y**2 + 22*z**2)) + 1j*-(x**6 + 2*y**6 - 23*y**4*z**2 + 8*y**2*z**4 - x**4*(8*y**2 + 7*z**2) + x**2*(-7*y**4 + 90*y**2*z**2 - 8*z**4)) )
            Bz = Bz * 9/8*rt35*( x*(x**2 - 3*y**2)*z*(3*x**2 + 3*y**2 - 8*z**2) + 1j*-y*(-3*x**2 + y**2)*z*(3*x**2 + 3*y**2 - 8*z**2) )
        elif m==4:
            rt35o2 = sqrt(35/2)
            Bx = Bx * 3*rt35o2*( 1/4*x*z*(7*x**4 + 23*y**4 + 12*y**2*z**2 - 2*x**2*(29*y**2 + 2*z**2)) + 1j*y*z*(8*x**4 + y**2*(y**2 + z**2) - x**2*(13*y**2 + 3*z**2)) )
            By = By * 3*rt35o2*( 1/4*y*z*(23*x**4 + 7*y**4 - 4*y**2*z**2 + x**2*(-58*y**2 + 12*z**2)) + 1j*-x*z*(x**4 + 8*y**4 - 3*y**2*z**2 + x**2*(-13*y**2 + z**2)) )
            Bz = Bz * -3*rt35o2*( 1/4*(x**4 - 6*x**2*y**2 + y**4)*(x**2 + y**2 - 10*z**2) + 1j*x*y*(x**2 - y**2)*(x**2 + y**2 - 10*z**2) )
        elif m==5:
            rt7 = sqrt(7)
            Bx = Bx * -3/8*rt7*( (6*x**6 - 5*y**4*(y**2 + z**2) - 5*x**4*(17*y**2 + z**2) + 10*x**2*(8*y**4 + 3*y**2*z**2)) + 1j*x*y*(35*x**4 + 31*y**4 + 20*y**2*z**2 - 10*x**2*(11*y**2 + 2*z**2)) )
            By = By * 3/8*rt7*( -x*y*(31*x**4 + 5*y**2*(7*y**2 - 4*z**2) + x**2*(-110*y**2 + 20*z**2)) + 1j*(5*x**6 - 6*y**6 + 5*y**4*z**2 + x**4*(-80*y**2 + 5*z**2) + 5*x**2*(17*y**4 - 6*y**2*z**2)) )
            Bz = Bz * -33/8*rt7*( x*(x**4 - 10*x**2*y**2 + 5*y**4)*z + 1j*y*(5*x**4 - 10*x**2*y**2 + y**4)*z )
        else:
            print(" m = ", m)
            raise ValueError(f"In field_xyz.eval_Bi, n={n} and m is not between -n and n.")

    elif n==6:
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #   n=6 moment components
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        A6 = sqrt(91/16/np.pi)
        Bx = A6*Binm/r**15
        By = Bx + 0.0
        Bz = Bx + 0.0

        if m==-6:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-5:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-4:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-3:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-2:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-1:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==0:
            Bx = Bx * ()
            By = By * ()
            Bz = Bz * ()
        elif m==1:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==2:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==3:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==4:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==5:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==6:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        else:
            print(" m = ", m)
            raise ValueError(f"In field_xyz.eval_Bi, n={n} and m is not between -n and n.")

    elif n==7:
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #   n=7 moment components
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        A7 = sqrt(15/16/np.pi)
        Bx = A7*Binm/r**17
        By = Bx + 0.0
        Bz = Bx + 0.0

        if m==-7:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-6:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-5:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-4:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-3:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-2:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-1:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==0:
            Bx = Bx * ()
            By = By * ()
            Bz = Bz * ()
        elif m==1:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==2:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==3:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==4:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==5:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==6:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==7:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        else:
            print(" m = ", m)
            raise ValueError(f"In field_xyz.eval_Bi, n={n} and m is not between -n and n.")

    elif n==8:
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #   n=8 moment components
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        A8 = 3/32*sqrt(17/2/np.pi)
        Bx = A8*Binm/r**19
        By = Bx + 0.0
        Bz = Bx + 0.0

        if m==-8:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-7:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-6:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-5:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-4:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-3:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-2:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-1:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==0:
            Bx = Bx * ()
            By = By * ()
            Bz = Bz * ()
        elif m==1:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==2:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==3:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==4:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==5:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==6:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==7:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==8:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        else:
            print(" m = ", m)
            raise ValueError(f"In field_xyz.eval_Bi, n={n} and m is not between -n and n.")

    elif n==9:
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #   n=9 moment components
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        A9 = sqrt(95/64/np.pi)
        Bx = A9*Binm/r**21
        By = Bx + 0.0
        Bz = Bx + 0.0

        if m==-9:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-8:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-7:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-6:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-5:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-4:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-3:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-2:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-1:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==0:
            Bx = Bx * ()
            By = By * ()
            Bz = Bz * ()
        elif m==1:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==2:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==3:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==4:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==5:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==6:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==7:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==8:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==9:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        else:
            print(" m = ", m)
            raise ValueError(f"In field_xyz.eval_Bi, n={n} and m is not between -n and n.")

    elif n==10:
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #   n=10 moment components
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        A10 = sqrt(1155/2/np.pi)/64
        Bx = A10*Binm/r**23
        By = Bx + 0.0
        Bz = Bx + 0.0

        if m==-10:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-9:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-8:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-7:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-6:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-5:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-4:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-3:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-2:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==-1:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==0:
            Bx = Bx * ()
            By = By * ()
            Bz = Bz * ()
        elif m==1:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==2:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==3:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==4:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==5:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==6:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==7:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==8:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==9:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        elif m==10:
            Bx = Bx * ( () + 1j*() )
            By = By * ( () + 1j*() )
            Bz = Bz * ( () + 1j*() )
        else:
            print(" m = ", m)
            raise ValueError(f"In field_xyz.eval_Bi, n={n} and m is not between -n and n.")

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
