import numpy as np
import pymech.exadata as exdat
import copy
from pymech.log import logger
#==============================================================================
def extrude(mesh, z, n, bc1, bc2, coord, ble, bte):
    """Extrudes a 2D mesh into a 3D one. Defines z positions and calls extrude_z
    
    Parameters
    ----------
    mesh : exadata
           2D mesh structure to extrude
    z : float
        list of z values of the boundaries of the intervals of the extruded mesh
    n : int
        list of number of elements per interval of the extruded mesh
    bc : str
         the boundary condition to use at the ends
    coord : float
            array of length four with the structure (xle,yle,xte,yte) where xle are the x and y coordinates of the leading edge. xte and yte are the counterparts for the trailing edge
    ble : float
          sweep angle (degrees) at the leading edge. A positive ble bends the mesh inward
    bte : float
          sweep angle (degrees) at the trailing edge. A positive bte bends the mesh inward (notice that it is opposite of aeronautical convention for wings)
    
    Special cases
    -------------
    1) n = [n{1},...,n{i},...,n{N-1}] and z = [z{1},...,z{i},...,z{N}] : The code extrudes the mesh between z{1} and z{N} with n{i} elements in the interval defined by z{i} and z{i+1} (len(n)=len(z)-1)
    2) n = [''] and z = [zmin,zmax] : The code extrudes the mesh between zmin and zmax with the normalized (between 0 and 1) point distribution from z.txt
    3) n = [dz0,s] and z = [zmin,zmax] : The code extrudes the mesh between zmin and zmax with a geometric point distribution defined by the initial spacing dz0 and inflation ratio s
    4) coord = [''] : xle/xte are set to the minimum/maximum x values of the mesh and yle/yte are set to the minimum/maximum y values of the mesh
    5) ble = -bte : sweep without taper not implemented
    """
    
    #Consistency checks: Initial grid
    if mesh.ndim != 2:
        logger.critical('The mesh to extrude must be 2D')
        return -1
    if mesh.lr1 != [2, 2, 1]:
        logger.critical('Only mesh structures can be extruded (lr1 = [2, 2, 1])')
        return -2
    if mesh.var[0] < 2:
        logger.critical('The mesh to extrude must contain (x, y) geometry')
        return -3
    if (bc1 == 'P' and bc2 != 'P') or (bc1 != 'P' and bc2 == 'P'):
        logger.critical('Inconsistent boundary conditions: one end is \'P\' but the other isn\'t')
        return -4
    
    if all(v=='' for v in n) and not any(v=='' for v in z) and len(z)==2:
        #print('Reading z-point distribution (scaled from 0 to 1) from z.txt file')
        f = open('z.txt')
        lines = f.readlines()
        f.close
        nz = len(lines)-1
        z1 = np.zeros((nz,1))
        z2 = np.zeros((nz,1))
        for k in range(0,nz):
            z1[k] = z[0]+float(lines[k])*(z[1]-z[0])
        for k in range(1,nz+1):
            z2[k-1] = z[0]+float(lines[k])*(z[1]-z[0])
        if len(z1) < 1:
            logger.critical('At least two points are necessary in the z.txt file')
            return -5
        #elif len(z1) == 1 or np.amax(np.diff(np.diff(np.append(z1,z2[nz-1])))) < 1e-15:
            #print('Uniform extrusion')
        #else:
            #print('Nonuniform extrusion') 
        mode = 0
    elif not any(v=='' for v in n) and not any(v=='' for v in z) and len(n)==len(z)-1:
        #print('Extrusion based on number of elements per interval and z coordinates of intervals')
        nz = np.sum(n)
        z1 = np.zeros((nz,1))
        z2 = np.zeros((nz,1))
        if len(z) < 2:
           logger.critical('z should contain at least two points')
           return -6
        #elif len(z) == 2 or np.amax(np.diff(np.divide(np.diff(z),n))) < 1e-15:
            #print('Uniform extrusion')
        #else:
            #print('Nonuniform extrusion')
        it = 0
        for kk in range(len(n)):
            for k in range(0,n[kk]):
                z1[it] = z[kk] + k/float(n[kk])*(z[kk+1]-z[kk])
                z2[it] = z[kk] + (k+1)/float(n[kk])*(z[kk+1]-z[kk])
                it += 1
        mode = 1
    elif not any(v=='' for v in n) and not any(v=='' for v in z) and len(n)==2 and len(z)==2:
        #print('Extrusion based on initial spacing and inflation ratio')
        #print('Nonuniform extrusion')
        dz0 = n[0]
        s = n[1]
        if s == 1:
            logger.critical('Inflation ratio cannot be 1')
            return -7
        nz = round(np.log(1.0-(z[1]-z[0])/dz0*(1.0-s))/np.log(s))
        z1 = np.zeros((nz,1))
        z2 = np.zeros((nz,1))
        z1[0] = z[0]
        z2[0] = z[0]+dz0
        for k in range(1,nz):
            z1[k] = z1[k-1]+dz0*s**(k-1)
            z2[k] = z2[k-1]+dz0*s**k
        if z2[-1]!=z[1]:
            print('Extrusion exceeds or does not reach requested zmax. Reducing or increasing z2 to zmax')
            z2[-1]=z[1]
            if abs(z1[-1])>=abs(z[1]):
                print('z1 needs to be reduced to be smaller than zmax')
                z1[-1]=0.5*(z1[-2]+z2[-1])
        mode = 2
    else: 
        logger.critical('Inconsistent n and z values')
        return -8
    
    #Consistency checks: z positions
    if not all(z1[v+1]==z2[v] for v in range(nz-1)):
        logger.critical('Inconsistent z values (check code)')
        return -9
    else: #Create list with all z positions
        zlist = np.zeros((nz+1,1))
        zlist[0:nz] = z1
        zlist[nz]   = z2[-1]
    
    mesh3d = extrude_z(mesh, zlist, bc1, bc2)
    mesh3d = taper(mesh3d, ble, bte, 0.0, coord, zlist[0])
    
    # return the extruded mesh
    return mesh3d

def extrude_z(mesh, z, bc1, bc2):
    """Extrudes a 2D mesh into a 3D one, given the z position of the nodes
    
    Parameters
    ----------
    mesh : exadata
           2D mesh structure to extrude
    z : float
        list of z values of the nodes of the elements of the extruded mesh
    bc : str
         the boundary condition to use at the ends
    
    Suggestion: see function extrude to understand how to call extrude_z
    """
    
    #Consistency checks: Initial grid
    if mesh.ndim != 2:
        logger.critical('The mesh to extrude must be 2D')
        return -1
    if mesh.lr1 != [2, 2, 1]:
        logger.critical('Only mesh structures can be extruded (lr1 = [2, 2, 1])')
        return -2
    if mesh.var[0] < 2:
        logger.critical('The mesh to extrude must contain (x, y) geometry')
        return -3
    if (bc1 == 'P' and bc2 != 'P') or (bc1 != 'P' and bc2 == 'P'):
        logger.critical('Inconsistent boundary conditions: one end is \'P\' but the other isn\'t')
        return -4
    
    nz=len(z)-1
    z1 = np.zeros((nz,1))
    z2 = np.zeros((nz,1))
    z1 = z[0:nz]
    z2 = z[1:nz+1]
    
    if (nz<=2 and bc1 == 'P'):
        print('Warning: Nek5000 has experienced errors in the past for n <= 2.') #version 17 of Nek5000 has had errors (Vanishing Jacobian errors) for 2 or less elements in the z direction and periodic boundary condition
    
    # copy the structure and make it 3D
    mesh3d = copy.deepcopy(mesh)
    mesh3d.lr1 = [2, 2, 2]
    mesh3d.var = [3, 0, 0, 0, 0]  # remove anything that isn't geometry for now
    nel2d = mesh.nel
    nel3d = mesh.nel*nz
    nbc = mesh.nbc
    mesh3d.nel = nel3d
    mesh3d.ndim = 3
    # The curved sides will also be extruded, one on each side of each element along nz
    mesh3d.ncurv = 2*nz*mesh.ncurv
    
    # add extra copies of all elements
    for k in range(nz-1):
        mesh3d.elem = mesh3d.elem + copy.deepcopy(mesh.elem)
    
    # set the z locations and curvature
    for k in range(nz):
        for i in range(nel2d):
            iel = i + nel2d*k
            # replace the position arrays with 3D ones (with z empty)
            mesh3d.elem[iel].pos = np.zeros((3, 2, 2, 2))
            mesh3d.elem[iel].pos[:, :1, :, :] = mesh.elem[i].pos
            mesh3d.elem[iel].pos[:, 1:2, :, :] = mesh.elem[i].pos

            # fill in the z location
            mesh3d.elem[iel].pos[2, 0, :, :] = z1[k]
            mesh3d.elem[iel].pos[2, 1, :, :] = z2[k]

            # extend curvature and correct it if necessary
            for icurv in range(4):
                curv_type = mesh3d.elem[iel].ccurv[icurv]
                # a 2D element has 4 edges that can be curved, numbered 0-3;
                # the extruded 3D element can have four more (on the other side), numbered 4-7
                mesh3d.elem[iel].ccurv[icurv+4] = curv_type
                mesh3d.elem[iel].curv[icurv+4] = mesh3d.elem[iel].curv[icurv]    # curvature params
                if curv_type == 'm':
                    # x and y are correct but z should be set to the proper value.
                    mesh3d.elem[iel].curv[icurv][2] = z1[k]
                    mesh3d.elem[iel].curv[icurv+4][2] = z2[k]
                elif curv_type == 's':
                    mesh3d.elem[iel].curv[icurv][2] = z1[k]
                    mesh3d.elem[iel].curv[icurv+4][2] = z2[k]
    
    # fix the internal boundary conditions (even though it's probably useless)
    # the end boundary conditions will be overwritten later with the proper ones
    for (iel, el) in enumerate(mesh3d.elem):
        for ibc in range(nbc):
            # set the conditions in faces normal to z
            el.bcs[ibc, 4][0] = 'E'
            el.bcs[ibc, 4][1] = iel+1
            el.bcs[ibc, 4][2] = 5
            el.bcs[ibc, 4][3] = iel-nel2d+1
            el.bcs[ibc, 4][4] = 6
            el.bcs[ibc, 5][0] = 'E'
            el.bcs[ibc, 5][1] = iel+1
            el.bcs[ibc, 5][2] = 6
            el.bcs[ibc, 5][3] = iel+nel2d+1
            el.bcs[ibc, 5][4] = 5
            # update the conditions for side faces
            for iface in range(4):
                el.bcs[ibc, iface][1] = iel+1
                if (el.bcs[ibc, iface][0] == 'E'):
                    # el.bcs[ibc, 0][1] ought to contain iel+1 once the mesh is valid
                    # but for now it should be off by a factor of nel2d because it is a copy of an element in the first slice
                    offset = iel-el.bcs[ibc, iface][1]+1
                    el.bcs[ibc, iface][3] = el.bcs[ibc, iface][3]+offset
    
    # now fix the end boundary conditions
    # face 5 is at zmin and face 6 is at zmax (with Nek indexing, corresponding to 4 and 5 in Python)
    for i in range(nel2d):
        for ibc in range(nbc):
            i1 = i+(nz-1)*nel2d  # index of the face on the zmax side
            mesh3d.elem[i].bcs[ibc, 4][0] = bc1
            mesh3d.elem[i].bcs[ibc, 4][1] = i+1
            mesh3d.elem[i].bcs[ibc, 4][2] = 5
            mesh3d.elem[i].bcs[ibc, 4][3] = 0.0
            mesh3d.elem[i].bcs[ibc, 4][4] = 0.0
            mesh3d.elem[i1].bcs[ibc, 5][0] = bc2
            mesh3d.elem[i1].bcs[ibc, 5][1] = i1+1
            mesh3d.elem[i1].bcs[ibc, 5][2] = 6
            mesh3d.elem[i1].bcs[ibc, 5][3] = 0.0
            mesh3d.elem[i1].bcs[ibc, 5][4] = 0.0
            # fix the matching faces for the periodic conditions
            if bc1 == 'P':
                mesh3d.elem[i].bcs[ibc, 4][3] = i1+1
                mesh3d.elem[i].bcs[ibc, 4][4] = 6
            if bc2 == 'P':
                mesh3d.elem[i1].bcs[ibc, 5][3] = i+1
                mesh3d.elem[i1].bcs[ibc, 5][4] = 5
    
    # return the extruded mesh
    return mesh3d

def taper(mesh, ble, bte, dih = 0.0, coord = [''], z0 = 0.0):
    """Modifies a 3D mesh (ideally extruded) to include taper, sweep and dihedral:
    
    Parameters
    ----------
    mesh : exadata
           3D mesh structure to modify
    ble : float
          sweep angle (degrees) at the leading edge. A positive ble bends the mesh inward
    bte : float
          sweep angle (degrees) at the trailing edge. A positive bte bends the mesh inward (notice that it is opposite of aeronautical convention for wings)
    dih : float
          dihedral angle (degrees) at the leading edge. A positive dih bends the mesh upward (default: 0.0, no dihedral)
    coord : float
            array of length four with the structure (xle,yle,xte,yte) where xle are the x and y coordinates of the leading edge. xte and yte are the counterparts for the trailing edge (default: [''], minimum/maximum x and y values - see special case 1)
    z0 : float
         reference z-position around which the taper, sweep and dihedral will be performed, the position which the mesh will not be scaled/deformed (default: 0.0)

    Special cases
    -------------
    1) coord = [''] : xle/xte are set to the minimum/maximum x values of the mesh and yle/yte are set to the minimum/maximum y values of the mesh
    2) ble = -bte : sweep without taper not implemented
    """
    if mesh.ndim != 3:
        logger.critical('The mesh to modified must be 3D')
        return -1
    
    nel=mesh.nel
    
    if all(v=='' for v in coord):
        #print('Leading/trailing edge is set to the minimum/maximum x,y-coordinates in the mesh')
        xvec = np.zeros((4*nel,1))
        yvec = np.zeros((4*nel,1))
        it = 0
        for iel in range(nel):
                for i in range(2):
                    for j in range(2):
                        xvec[it] = mesh.elem[iel].pos[0,0,j,i]
                        yvec[it] = mesh.elem[iel].pos[1,0,j,i]
                        it += 1
        indxmin = np.argmin(xvec)
        indxmax = np.argmax(xvec)
        indymin = np.argmin(yvec)
        indymax = np.argmax(yvec)
        xle = xvec[indxmin]
        yle = yvec[indymin]
        xte = xvec[indxmax]
        yte = yvec[indymax]
        zle = z0
        zte = z0
    elif not any(v=='' for v in coord):
        #print('Leading/trailing edge is set to the x,y-coordinates from the user`s definition')
        xle = coord[0]
        yle = coord[1]
        xte = coord[2]
        yte = coord[3]
        zle = z0
        zte = z0
    else:
        logger.critical('The x and y positions of the leading and trailing edges are inconsistent')
        return -9
    
    ble = np.deg2rad(ble)
    bte = np.deg2rad(bte)
    dih = np.deg2rad(dih)
    x1 = np.zeros((2,2,2))
    y1 = np.zeros((2,2,2))
    
    if np.tan(ble)+np.tan(bte) == 0.0:
        if (ble==0.0 and bte==0.0):
            print('No change to mesh inside taper')
            return mesh
        else:
            logger.critical('Sweep without taper not implemented')
            return -2
    else:
        #coordinates of apex of auxiliar triangle
        xa = xle+np.tan(ble)*(xte-xle+np.tan(bte)*zte-np.tan(bte)*zle)/(np.tan(ble)+np.tan(bte))
        za = (xte-xle+np.tan(bte)*zte+np.tan(ble)*zle)/(np.tan(ble)+np.tan(bte))
        ya = 0.5*(yle+yte)+np.tan(dih)*(za-z0)
    
    for (iel, el) in enumerate(mesh.elem):
        # scale and translate x and y coordinates if nonorthogonal extrusion
        x1 = xa+(xa-el.pos[0, :, :, :])/(za-z0)*(el.pos[2, :, :, :]-za)
        y1 = ya+(ya-el.pos[1, :, :, :])/(za-z0)*(el.pos[2, :, :, :]-za)
        el.pos[0, :, :, :] = x1
        el.pos[1, :, :, :] = y1
        for icurv in range(12):
            curv_type = el.ccurv[icurv]
            if curv_type == 'm':
                # scale and translate x and y coordinates if nonorthogonal extrusion
                el.curv[icurv][0] = xa+(xa-el.curv[icurv][0])/(za-z0)*(el.curv[icurv][2]-za)
                el.curv[icurv][1] = ya+(ya-el.curv[icurv][1])/(za-z0)*(el.curv[icurv][2]-za)
            elif curv_type == 'C':
                #find z-position of edge, to rescale radius
                if icurv == 0:
                    zr=0.5*(el.pos[2, 0, 0, 0]+el.pos[2, 0, 0, 1])
                elif icurv == 1:
                    zr=0.5*(el.pos[2, 0, 0, 1]+el.pos[2, 0, 1, 1])
                elif icurv == 2:
                    zr=0.5*(el.pos[2, 0, 1, 1]+el.pos[2, 0, 1, 0])
                elif icurv == 3:
                    zr=0.5*(el.pos[2, 0, 1, 0]+el.pos[2, 0, 0, 0])
                elif icurv == 4:
                    zr=0.5*(el.pos[2, 1, 0, 0]+el.pos[2, 1, 0, 1])
                elif icurv == 5:
                    zr=0.5*(el.pos[2, 1, 0, 1]+el.pos[2, 1, 1, 1])
                elif icurv == 6:
                    zr=0.5*(el.pos[2, 1, 1, 1]+el.pos[2, 1, 1, 0])
                elif icurv == 7:
                    zr=0.5*(el.pos[2, 1, 1, 0]+el.pos[2, 1, 0, 0])
                else:
                    print('Warning: Curvature C is not defined for edges 9-12. Nek5000 will ignore it. Ignoring in taper.')
                # scaling the radius of the circle
                el.curv[icurv][0] = el.curv[icurv][0]*abs((zr-za)/(za-z0))
            else:
                print('Warning: Taper only implemented only for curvatures m and C. Curvature s acts on faces, not consistent with taper.')
    
    # return the modified mesh
    return mesh

def edge_mid(el, iedge):
    """Finds the coordinates of the midsize-node of edge iedge of element el (in other words, if the curvature were type 'm', the values of el.curv[iedge][:3]):
    
    Parameters
    ----------
    el : exadata
         element of mesh (usually, el=mesh.elem[i])
    iedge : int
            index of edge
    """
    
    #correct if ccurv=='m', otherwise, works as allocation
    midpoint=copy.deepcopy(el.curv[iedge][:3])
    
    if el.ccurv[iedge] != 'm':
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
        elif iedge == 4:
            pos1 = el.pos[:, 1, 0, 0]
            pos2 = el.pos[:, 1, 0, 1]
        elif iedge == 5:
            pos1 = el.pos[:, 1, 0, 1]
            pos2 = el.pos[:, 1, 1, 1]
        elif iedge == 6:
            pos1 = el.pos[:, 1, 1, 1]
            pos2 = el.pos[:, 1, 1, 0]
        elif iedge == 7:
            pos1 = el.pos[:, 1, 1, 0]
            pos2 = el.pos[:, 1, 0, 0]
        elif iedge == 8:
            pos1 = el.pos[:, 0, 0, 0]
            pos2 = el.pos[:, 1, 0, 0]
        elif iedge == 9:
            pos1 = el.pos[:, 0, 0, 1]
            pos2 = el.pos[:, 1, 0, 1]
        elif iedge == 10:
            pos1 = el.pos[:, 0, 1, 1]
            pos2 = el.pos[:, 1, 1, 1]
        elif iedge == 11:
            pos1 = el.pos[:, 0, 1, 0]
            pos2 = el.pos[:, 1, 1, 0]
        
        if el.ccurv[iedge] == '':
            midpoint = (pos1+pos2)/2.0
        elif el.ccurv[iedge] == 'C':
            # Curvature 'C' only needs x and y. Works for 2d and extruded meshes.
            if iedge > 7:
                # For iedge=8-11: will give a different value to what Nek considers (Nek ignores it).
                print('Calculating midpoint differently from Nek5000. Nek ignores it for edges 9-12.')
            radius = el.curv[iedge][0]
            dmid = abs(radius)-(radius**2-(pos2[0]-pos1[0])**2/4.0-(pos2[1]-pos1[1])**2/4.0)**0.5
            midpoint[0] = (pos2[0]+pos1[0])/2.0+dmid/((pos2[0]-pos1[0])**2+(pos2[1]-pos1[1])**2)**(0.5)*radius/abs(radius)*(pos2[1]-pos1[1])
            midpoint[1] = (pos2[1]+pos1[1])/2.0-dmid/((pos2[0]-pos1[0])**2+(pos2[1]-pos1[1])**2)**(0.5)*radius/abs(radius)*(pos2[0]-pos1[0])
            midpoint[2] = (pos2[2]+pos1[2])/2.0
        elif el.ccurv[iedge] == 's':
            # It doesn't check if sphere is consistent with pos1 and pos2. Just assumes it is.
            radius = el.curv[iedge][4]
            center = el.curv[iedge][:3]
            dist = (pos2+pos1)/2.0-center
            midpoint = center + dist*radius/(dist[0]**2+dist[1]**2+dist[2]**2)**0.5
    
    # return the coordinate of midsize node
    return midpoint

def edge_circle(el, iedge, midpoint):
    """Finds the radius of curvature and circle center based on the midsize-node of edge iedge of element el:
    
    Parameters
    ----------
    el : exadata
         element of mesh (usually, el=mesh.elem[i])
    iedge : int
            index of edge
    midpoint : float
               list of coordinates of midsize-node (in other words, if the curvature were type 'm', the values of el.curv[iedge][:3])
    """
    
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
    elif iedge == 4:
        pos1 = el.pos[:, 1, 0, 0]
        pos2 = el.pos[:, 1, 0, 1]
    elif iedge == 5:
        pos1 = el.pos[:, 1, 0, 1]
        pos2 = el.pos[:, 1, 1, 1]
    elif iedge == 6:
        pos1 = el.pos[:, 1, 1, 1]
        pos2 = el.pos[:, 1, 1, 0]
    elif iedge == 7:
        pos1 = el.pos[:, 1, 1, 0]
        pos2 = el.pos[:, 1, 0, 0]
    elif iedge == 8:
        pos1 = el.pos[:, 0, 0, 0]
        pos2 = el.pos[:, 1, 0, 0]
    elif iedge == 9:
        pos1 = el.pos[:, 0, 0, 1]
        pos2 = el.pos[:, 1, 0, 1]
    elif iedge == 10:
        pos1 = el.pos[:, 0, 1, 1]
        pos2 = el.pos[:, 1, 1, 1]
    elif iedge == 11:
        pos1 = el.pos[:, 0, 1, 0]
        pos2 = el.pos[:, 1, 1, 0]
    
    side1 = midpoint-pos1
    side2 = pos2-midpoint
    side3 = pos1-pos2
    
    d1 = (side1[0]**2+side1[1]**2+side1[2]**2)**0.5
    d2 = (side2[0]**2+side2[1]**2+side2[2]**2)**0.5
    d3 = (side3[0]**2+side3[1]**2+side3[2]**2)**0.5
    sper = (d1+d2+d3)/2.0
    area = (sper*(sper-d1)*(sper-d2)*(sper-d3))**0.5
    
    if area > 0.0001*d1*d2:
        radius = d1*d2*d3/(4*area)
        alpha1 = d2**2*(d1**2+d3**2-d2**2)/2.0
        alpha2 = d3**2*(d2**2+d1**2-d3**2)/2.0
        alpha3 = d1**2*(d3**2+d2**2-d1**2)/2.0
        center = (alpha1*pos1+alpha2*midpoint+alpha3*pos2)/(8.0*area**2)
        if ((side1[0]-side3[0])*(side2[1]-side1[1])-(side1[0]-side2[0])*(side3[1]-side1[1])) < 0.0:
            # if curvature == 'C', the radius is negative for clockwise triangles 
            # works only for 2d/extruded mesh - do not know how to interpret it in 3d (should work for edges edges 0-7 of extruded meshes, unknown behaviour for edges 8-11)
            radius = -radius
    else:
        #radius is too big compared to edge. For pratical purposes, no curvature: radius=0.0
        radius = 0.0
        center = [0.0, 0.0, 0.0]
    
    curv = copy.deepcopy(el.curv[iedge][:4])
    curv[0] = radius
    curv[1:4] = center
    
    # return radius and center of curvature (it is not ready to be used: For 'C', define curv[1:4]=0.0; For 's', define el.curv[4]=abs(curv[0]) and el.curv[:3]=curv[1:4])
    return curv
