import math
from math import pi,cos,sin,tan,atan2

def fun_circle(xpos, ypos, Rlim):
    """
    Function to define high discretization mesh and low discretization mesh.
    Given x and y-position (xpos, ypos) of a point of a 2d mesh, funval can be positive or negative
    Internal mesh: funval should be negative. External mesh: funval should be positive (the definition of internal/external is interchangeable, but they must be consistent if multiple functions are defined).
    The boundary defined by this function should cross the mid elements such that exactly 2 points are in each side of the boundary (2 points inside and 2 points are outside).
    For fun_circle, the boundary is a circle of radius Rlim.
    ----------
    """
    funval = ((xpos**2+ypos**2)**0.5)/Rlim - 1.0
    return funval

def fun_hexag(xpos, ypos, Rlim):
    """
    Function to define high discretization mesh and low discretization mesh.
    Given x and y-position (xpos, ypos) of a point of a 2d mesh, funval can be positive or negative
    Internal mesh: funval should be negative. External mesh: funval should be positive (the definition of internal/external is interchangeable, but they must be consistent if multiple functions are defined).
    The boundary defined by this function should cross the mid elements such that exactly 2 points are in each side of the boundary (2 points inside and 2 points are outside).
    For fun_hexag, the boundary is a hexagon of apothem Rlim (probably there is a smarter way to do this, but it does the job).
    ----------
    """
    theta = atan2(ypos, xpos)
    if theta > pi-pi/6.0 or theta <= -(pi-pi/6.0):
        funval = -(xpos + Rlim)
    elif theta > pi/2.0:
        xaux = (xpos-Rlim*cos(pi-pi/3.0)) 
        yaux = (ypos-Rlim*sin(pi-pi/3.0))
        thetaaux = atan2(yaux, xaux)
        if thetaaux < -(pi-pi/6.0):
            funval=+1.0
        elif thetaaux < pi/6.0:
            funval=-1.0
        else:
            funval=+1.0
    elif theta > pi/6.0:
        xaux = (xpos-Rlim*cos(pi/3.0)) 
        yaux = (ypos-Rlim*sin(pi/3.0))
        thetaaux = atan2(yaux, xaux)
        if thetaaux < -pi/6.0:
            funval=-1.0
        elif thetaaux < pi-pi/6.0:
            funval=+1.0
        else:
            funval=-1.0
    elif theta > -pi/6.0:
        funval = (xpos - Rlim)
    elif theta > -pi/2.0:
        xaux = (xpos-Rlim*cos(-pi/3.0)) 
        yaux = (ypos-Rlim*sin(-pi/3.0))
        thetaaux = atan2(yaux, xaux)
        if thetaaux < -(pi-pi/6.0):
            funval=-1.0
        elif thetaaux < pi/6.0:
            funval=+1.0
        else:
            funval=-1.0
    elif theta <= -pi/2.0:
        xaux = (xpos-Rlim*cos(-(pi-pi/3.0))) 
        yaux = (ypos-Rlim*sin(-(pi-pi/3.0)))
        thetaaux = atan2(yaux, xaux)
        if thetaaux < -pi/6.0:
            funval=+1.0
        elif thetaaux < pi-pi/6.0:
            funval=-1.0
        else:
            funval=+1.0
    return funval
