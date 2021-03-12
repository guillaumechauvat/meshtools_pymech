import numpy as np
import pymech.neksuite as ns
import pymech.exadata as exa
import meshtools_split as msts
import copy
import matplotlib.path as mpltPath
from numpy import pi,cos,sin,tan,arctan2
from pymech.log import logger

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

def fun_polyg(xpos, ypos, xyzline):
    """
    Function to define high discretization mesh and low discretization mesh.
    Given x and y-position (xpos, ypos) of a point of a 2d mesh, funval can be positive or negative
    Internal mesh: funval should be negative. External mesh: funval should be positive (the definition of internal/external is interchangeable, but they must be consistent if multiple functions are defined).
    The boundary defined by this function should cross the mid elements such that exactly 2 points are in each side of the boundary (2 points inside and 2 points are outside).
    For fun_polyg, the boundary is a polygon of any form formed by the positions xyzline (xyzline may be x,y,z or just x,y).
    ----------
    """
    
    xyzline = xyzline[:,:2]
    xypos= np.zeros((1,2))
    xypos[0,0] = xpos
    xypos[0,1] = ypos
    
    path = mpltPath.Path(xyzline)
    inside = path.contains_points(xypos)
    
    if inside:
        funval = -1.0
    else:
        funval = 1.0
    
    return funval

def lim_polyg(mesh, iel0, iedge0):
    """
    Function to define line that intersect all neighboring elements starting from a element iel and edge iedge. It defines a polygon using the midpoints of the intersected edges.
    Keep in mind: if the elements do not form a closed polygon, the user needs to add points to array xyzline (x,y,z of vertices of polygon) by hand to close the polygon.
    Used to define array that is input for fun_polyg.
    ----------
    """
    xyzline = np.empty((0,3))
    xyzline2 = np.empty((0,3))
    
    #iel0 = iel0-1 #transforming Nek index to pymech index
    #iedge0 = iedge0-1 #transforming Nek index to pymech index
    
    iel = iel0+1
    iedge = iedge0
    
    el = mesh.elem[iel0]
    
    #xyzline1 = [msts.edge_mid(mesh.elem[iel0], iedge0)]
    #iedge = (iedge+2)%4
    iedge = iedge0
    while (el.bcs[0, iedge][0] == 'E' and iel !=iel0):
        xyzline = np.append(xyzline, [msts.edge_mid(el, iedge)], axis=0)
        iel = int(el.bcs[0, iedge][3]-1)
        iedge = int(el.bcs[0, iedge][4]-1)
        el = mesh.elem[iel]
        iedge = (iedge+2)%4
    xyzline = np.append(xyzline, [msts.edge_mid(el, iedge)], axis=0)
    
    if iel !=iel0:
        el = mesh.elem[iel0]
        iedge = iedge0
        iedge = (iedge0+2)%4
        while (el.bcs[0, iedge][0] == 'E' and iel !=iel0):
            xyzline2 = np.append(xyzline2, [msts.edge_mid(el, iedge)], axis=0)
            iel = int(el.bcs[0, iedge][3]-1)
            iedge = int(el.bcs[0, iedge][4]-1)
            el = mesh.elem[iel]
            iedge = (iedge+2)%4
        xyzline2 = np.append(xyzline2, [msts.edge_mid(el, iedge)], axis=0)
        xyzline = np.append(np.flip(xyzline, axis=0), xyzline2, axis=0)
    
    return xyzline

def iel_point(mesh, xyzpoint):
    """
    Function to find which element contains point defined in xypoint (does not consider curvature, treats elements as quadrilateral)
    ----------
    """
    #Consistency checks: Initial grid
    if mesh.ndim != 2:
        print('Warning: mesh must be 2D to find which element contains the point.')
    
    ifound = 0
    area = np.zeros((4))
    for (iel, el) in enumerate(mesh.elem):
        for iedge in range(4):
            if iedge == 0:
                pos1 = el.pos[:, 0, 0, 0]
                pos2 = el.pos[:, 0, 0, 1]
            elif iedge == 1:
                pos1 = el.pos[:, 0, 0, 1]
                pos2 = el.pos[:, 0, 1, 1]
            elif iedge == 2:
                pos1 = el.pos[:, 0, 1, 1]
                pos2 = el.pos[:, 0, 1, 0]
            elif iedge == 3:
                pos1 = el.pos[:, 0, 1, 0]
                pos2 = el.pos[:, 0, 0, 0]
                
            side1 = xyzpoint-pos1
            side2 = pos2-xyzpoint
            side3 = pos1-pos2
            #Calculating signed area of triangle
            area[iedge] = 0.5*((side1[0]-side3[0])*(side2[1]-side1[1])-(side1[0]-side2[0])*(side3[1]-side1[1]))
        #If both areas with oposing edges have the same sign, it is in the inside the element
        if (area[0]*area[2] > 0.0 and area[1]*area[3] > 0.0):
            ielpoint = iel
            if ifound > 0:
                print('Warning: point is inside more than one element.')
            ifound = ifound+1
    
    return ielpoint

def iface_neig(mesh, iel, ielneig):
    """
    Function to find which face of mesh.elem[iel] is connected to mesh.elem[ielneig]
    ----------
    """
    
    el = mesh.elem[iel]
    ifaceneig = ''
    
    for iface in range(6):
        if (el.bcs[0, iface][0] == 'E') and (ielneig == int(el.bcs[0, iface][3]-1)):
            ifaceneig = int(el.bcs[0, iface][4]-1)
    
    if ifaceneig == '':
        logger.critical('Neighboring face was not found. Elements are not neighbours or bc not equal to E')
        return ''
    
    return ifaceneig

def fun_hexag(xpos, ypos, Rlim):
    """
    Function to define high discretization mesh and low discretization mesh.
    Given x and y-position (xpos, ypos) of a point of a 2d mesh, funval can be positive or negative
    Internal mesh: funval should be negative. External mesh: funval should be positive (the definition of internal/external is interchangeable, but they must be consistent if multiple functions are defined).
    The boundary defined by this function should cross the mid elements such that exactly 2 points are in each side of the boundary (2 points inside and 2 points are outside).
    For fun_hexag, the boundary is a hexagon of apothem Rlim (there is a smarter way to do this, but it does the job).
    ----------
    """
    theta = arctan2(ypos, xpos)
    if theta > pi-pi/6.0 or theta <= -(pi-pi/6.0):
        funval = -(xpos + Rlim)
    elif theta > pi/2.0:
        xaux = (xpos-Rlim*cos(pi-pi/3.0)) 
        yaux = (ypos-Rlim*sin(pi-pi/3.0))
        thetaaux = arctan2(yaux, xaux)
        if thetaaux < -(pi-pi/6.0):
            funval=+1.0
        elif thetaaux < pi/6.0:
            funval=-1.0
        else:
            funval=+1.0
    elif theta > pi/6.0:
        xaux = (xpos-Rlim*cos(pi/3.0)) 
        yaux = (ypos-Rlim*sin(pi/3.0))
        thetaaux = arctan2(yaux, xaux)
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
        thetaaux = arctan2(yaux, xaux)
        if thetaaux < -(pi-pi/6.0):
            funval=-1.0
        elif thetaaux < pi/6.0:
            funval=+1.0
        else:
            funval=-1.0
    elif theta <= -pi/2.0:
        xaux = (xpos-Rlim*cos(-(pi-pi/3.0))) 
        yaux = (ypos-Rlim*sin(-(pi-pi/3.0)))
        thetaaux = arctan2(yaux, xaux)
        if thetaaux < -pi/6.0:
            funval=+1.0
        elif thetaaux < pi-pi/6.0:
            funval=-1.0
        else:
            funval=+1.0
    return funval
