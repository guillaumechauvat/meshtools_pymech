import numpy as np
import pymech.exadata as exdat
import copy
import meshtools8 as mst
from pymech.log import logger
#==============================================================================
def extrude_split(mesh2D, z, n, bc1, bc2, fun  = '', Rlim = '', imesh_high = 0):
    """Extrudes a 2D mesh into a 3D one, following the pattern:
     _____ _____ _____ _____
    |     |     |     |     |
    |     |     |     |     |
    |     |     |     |     |
    |_____|_____|_____|_____|
    |    /|\    |    /|\    |
    |__ / | \ __|__ / | \ __| (fun (with parameter Rlim) should change change sign in the mid element)
    |  |  |  |  |  |  |  |  | (half of the mid elements are also divided in 2 in (x,y)-plane)
    |__|__|__|__|__|__|__|__|
    |  |  |  |  |  |  |  |  |
    |  |  |  |  |  |  |  |  |
    |  |  |  |  |  |  |  |  | (imesh_high is the index of mesh with higher intended discretization in z)
    |__|__|__|__|__|__|__|__|
    
    The pattern is similar to "Picture Frame" of an ancient NEKTON manual (https://www.mcs.anl.gov/~fischer/Nek5000/nekmanual.pdf).
    If the mid elements have curvature, the extrusion might modify it. Do not split in regions where the value of curvature parameters is very important.
    
    Parameters
    ----------
    mesh2D : exadata
           2D mesh structure to extrude
    z : float
        list of z values of the boundaries of the intervals of the extruded mesh
    n : int
        list of number of elements per interval of the extruded mesh
    bc : str
         the boundary condition to use at the ends
    fun: function
         list of functions that define the splitting lines for different discretization meshes (default: empty, calls simple extrusion routine)
    Rlim: float
          list of parameters for functions that define the splitting lines for different discretization meshes (default: empty, for when Rlim is not needed inside fun)
    imesh_high : int
                 index of fun that defines the mesh with higher discretization. Example: 0, is the most internal mesh; 1 is the second most internal mesh, etc (default: the most internal mesh, imesh_high=0)

    Special cases
    -------------
    1) n = [n{1},...,n{i},...,n{N-1}] and z = [z{1},...,z{i},...,z{N}] : The code extrudes the mesh between z{1} and z{N} with n{i} elements in the interval defined by z{i} and z{i+1} (len(n)=len(z)-1)
    2) n = [''] and z = [zmin,zmax] : The code extrudes the mesh between zmin and zmax with the normalized (between 0 and 1) point distribution from z.txt
    3) n = [dz0,s] and z = [zmin,zmax] : The code extrudes the mesh between zmin and zmax with a geometric point distribution defined by the initial spacing dz0 and inflation ratio s
    4) fun = '' : The code ignores the splitting part and calls simple extrusion routine. Observe 'return mesh3D' right after definition of z.
    """
    
    #Consistency checks: Initial grid
    if mesh2D.ndim != 2:
        logger.critical('The mesh to extrude must be 2D')
        return -1
    if mesh2D.lr1 != [2, 2, 1]:
        logger.critical('Only mesh structures can be extruded (lr1 = [2, 2, 1])')
        return -2
    if mesh2D.var[0] < 2:
        logger.critical('The mesh to extrude must contain (x, y) geometry')
        return -3
    #Consistency checks: Periodic boundary condition
    if (bc1 == 'P' and bc2 != 'P') or (bc1 != 'P' and bc2 == 'P'):
        logger.critical('Inconsistent boundary conditions: one end is \'P\' but the other isn\'t')
        return -4
    
    #Consistency checks: Functions that define the splitting lines
    nsplit=len(fun)
    if len(Rlim) < nsplit:
        print('Warning: Length of Rlim < lenght of fun. Completing with', (nsplit-len(Rlim)), 'zeros.')
        Rlim[len(Rlim):nsplit]=[0.0]*(nsplit-len(Rlim))
    elif len(Rlim) > nsplit:
        print('Warning: Length of Rlim > lenght of fun. Ignoring', (len(Rlim)-nsplit), 'values.')
    
    #Defining z positions for extrusion
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
            z2[-1] = z[1]
            if abs(z1[-1])>=abs(z[1]):
                print('z1 needs to be reduced to be smaller than zmax')
                z1[-1] = 0.5*(z1[-2]+z2[-1])
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
    #Consistency checks: if nz is divided by 4 (or 8, or 16, etc)
    if (nz % 2**abs(imesh_high+1) != 0) or (nz % 2**abs(nsplit-imesh_high+1) != 0):
        logger.critical(f'Inconsistent elements to extrude: n must be divided by {max([2**abs(imesh_high+1),2**abs(nsplit-imesh_high+1)])}')
        return -10
    
    #If fun is not defined, there is no splitting to be done. Call simple extrusion and end routine
    if fun == '':
        print('Splitting function not defined. Calling simple extrusion routine.')
        mesh3D=mst.extrude_z(mesh2D, zlist, bc1, bc2)
        return mesh3D
    
    mesh2D_ext = copy.deepcopy(mesh2D)
    meshes2D=[] #list of 2D meshes
    meshes3D=[] #list of 3D meshes
    
    #Splitting 2D meshes
    print('Splitting 2D meshes. Make sure splitting is done where value of curvature is not important.')
    for k in range(nsplit):
        meshes2D.append(copy.deepcopy(mesh2D_ext))
        meshes2D.append(copy.deepcopy(mesh2D_ext))
        
        iel_int = 0
        iel_mid = 0
        iel_ext = 0
        
        for iel in range(mesh2D_ext.nel):
            it = 0
            xvec = np.zeros((4,1))
            yvec = np.zeros((4,1))
            rvec = np.zeros((4,1))
            for i in range(2):
                for j in range(2):
                    xvec[it] = mesh2D_ext.elem[iel].pos[0,0,j,i]
                    yvec[it] = mesh2D_ext.elem[iel].pos[1,0,j,i]
                    rvec[it] = fun[k](xvec[it], yvec[it], Rlim[k])
                    it += 1
            if max(rvec) <= 0.0:
                meshes2D[2*k].elem[iel_int] = copy.deepcopy(mesh2D_ext.elem[iel])
                iel_int += 1
            elif min(rvec) > 0.0:
                mesh2D_ext.elem[iel_ext] = copy.deepcopy(mesh2D_ext.elem[iel])
                iel_ext += 1
            else:
                meshes2D[2*k+1].elem[iel_mid] = copy.deepcopy(mesh2D_ext.elem[iel])
                iel_mid += 1
                        
        meshes2D[2*k].nel=iel_int
        meshes2D[2*k+1].nel=iel_mid
        mesh2D_ext.nel=iel_ext
        
        print('Mesh2D', 2*k, 'elements:', meshes2D[2*k].nel)
        print('Mesh2D', 2*k+1, 'elements:', meshes2D[2*k+1].nel)
        
        meshes2D[2*k].elem=meshes2D[2*k].elem[:iel_int]
        meshes2D[2*k+1].elem=meshes2D[2*k+1].elem[:iel_mid]
        mesh2D_ext.elem=mesh2D_ext.elem[:iel_ext]
        
        ncurv = 0
        for el in meshes2D[2*k].elem:
            for iedge in range(12):
                if el.ccurv[iedge] != '':
                    ncurv = ncurv+1
        meshes2D[2*k].ncurv = ncurv
                    
        ncurv = 0
        for el in meshes2D[2*k+1].elem:
            for iedge in range(12):
                if el.ccurv[iedge] != '':
                    ncurv = ncurv+1
        meshes2D[2*k+1].ncurv = ncurv
    
    #End of splitting, remaining is the last mesh: Mesh_ext
    print('Mesh2Dext elements:', mesh2D_ext.nel)
    ncurv = 0
    for el in mesh2D_ext.elem:
        for iedge in range(12):
            if el.ccurv[iedge] != '':
                ncurv = ncurv+1
    mesh2D_ext.ncurv = ncurv
    
    #Extruding meshes
    print('Extruding meshes')
    for k in range(nsplit):
        n_local = [int(x/2**abs(k-imesh_high)) for x in n]
        zlist_local=zlist[::int(2**abs(k-imesh_high))]
        
        if k<imesh_high:
            fun_local = lambda xpos, ypos, rlim : -fun[k](xpos,ypos,rlim)
            n_mid = [int(2*x) for x in n_local]
            zlist_mid=zlist[::int(2**abs(k-imesh_high+1))]
        else:
            fun_local = fun[k]
            n_mid = n_local
            zlist_mid=zlist_local
        
        for x in n_mid:
            if x % 4 != 0:
                logger.critical('Inconsistent elements to extrude: n must be divided by 4')
                return -11
        
        meshes3D.append(mst.extrude_z(meshes2D[2*k], zlist_local, bc1, bc2))
        meshes3D.append(extrude_mid(meshes2D[2*k+1], zlist_mid, bc1, bc2, fun_local, Rlim[k]))
        
        #print('Removing boundary condition E to improve merging time')
        for (iel, el) in enumerate(meshes3D[2*k].elem):
            for ibc in range(meshes3D[2*k].nbc):
                for iface in range(6):
                    el.bcs[ibc, iface][1] = iel+1
                    if el.bcs[ibc, iface][0] == 'E':
                        el.bcs[ibc, iface][0] = ''
                        el.bcs[ibc, iface][1] = iel+1
                        el.bcs[ibc, iface][2] = iface
                        el.bcs[ibc, iface][3] = 0.0
                        el.bcs[ibc, iface][4] = 0.0
        
        for (iel, el) in enumerate(meshes3D[2*k+1].elem):
            for ibc in range(meshes3D[2*k+1].nbc):
                for iface in range(6):
                    el.bcs[ibc, iface][1] = iel+1
                    if el.bcs[ibc, iface][0] == 'E':
                        el.bcs[ibc, iface][0] = ''
                        el.bcs[ibc, iface][1] = iel+1
                        el.bcs[ibc, iface][2] = iface
                        el.bcs[ibc, iface][3] = 0.0
                        el.bcs[ibc, iface][4] = 0.0
        
        print('Mesh3D', 2*k, 'elements:', meshes3D[2*k].nel)
        print('Mesh3D', 2*k+1, 'elements:', meshes3D[2*k+1].nel)
    
    n_local = [int(x/2**abs(nsplit-imesh_high)) for x in n]
    zlist_local=zlist[::int(2**abs(nsplit-imesh_high))]
    mesh3D_ext = mst.extrude_z(mesh2D_ext, zlist_local, bc1, bc2)
    for (iel, el) in enumerate(mesh3D_ext.elem):
        for ibc in range(mesh3D_ext.nbc):
            for iface in range(6):
                el.bcs[ibc, iface][1] = iel+1
                if el.bcs[ibc, iface][0] == 'E':
                    el.bcs[ibc, iface][0] = ''
                    el.bcs[ibc, iface][1] = iel+1
                    el.bcs[ibc, iface][2] = iface
                    el.bcs[ibc, iface][3] = 0.0
                    el.bcs[ibc, iface][4] = 0.0
    
    print('Mesh3Dext elements:', mesh3D_ext.nel)
    
    #Merging meshes
    print('Merging meshes')
    print('Boundary condition E were removed to improve merging time')
    for k in range(nsplit):
        if k==0:
            mesh3D = copy.deepcopy(meshes3D[2*k])
        else:
            mesh3D = merge(mesh3D,meshes3D[2*k])
        #print('Mesh', 2*k, 'merged. Elements:', mesh3D.nel)
        mesh3D = merge(mesh3D,meshes3D[2*k+1])
        #print('Mesh', 2*k+1, 'merged. Elements:', mesh3D.nel)
    
    mesh3D = merge(mesh3D,mesh3D_ext)
    #print('Mesh ext merged. Elements:', mesh3D.nel)
    print('Merging done. Total elements:', mesh3D.nel)
    
    ncurv = 0
    for el in mesh3D.elem:
        for iedge in range(12):
            if el.ccurv[iedge] != '':
                ncurv = ncurv+1
    mesh3D.ncurv = ncurv
    
    # return the extruded mesh
    return mesh3D
        
def extrude_mid(mesh, z, bc1, bc2, fun, Rlim = 0.0):
    """Extrudes the mid elments of the 2D mesh into a 3D one. Following the pattern:
     _____ _____ _____ _____ 
    |1   /|\   4|    /|\    |
    |__ / | \ __|__ / | \ __| (fun (with parameter Rlim) should change change sign in the mid element)
    |0 |2 |3 | 5|  |  |  |  | (half of the mid elements are also divided in 2 in (x,y)-plane)
    |__|__|__|__|__|__|__|__| (numbers in the figure indicate the indices (iel+0; iel+1; etc))
    
    Parameters
    ----------
    mesh : exadata
           2D mesh structure to extrude
    z : float
        list of z values of the nodes of the elements of the extruded mesh in the high discretization region (len(z)-1 must be divide by 4)
    bc : str
         the boundary condition to use at the ends
    fun : function
          function that define the splitting lines for different discretization meshes
    Rlim : float
           parameter for functions that define the splitting lines for different discretization meshes (default: zero, can be used for when Rlim is not needed inside fun)

    Suggestion: see function extrude_split to understand how to call extrude_mid
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
    
    if nz % 4 != 0:
        logger.critical('Inconsistent elements to extrude: nz must be divided by 4')
        return -5
 
    # copy the structure and make it 3D
    mesh3d = copy.deepcopy(mesh)
    mesh3d.lr1 = [2, 2, 2]
    mesh3d.var = [3, 0, 0, 0, 0]  # remove anything that isn't geometry for now
    nel2d = mesh.nel
    nel3d = nel2d*6*int(nz/4) # every mid-extrusion of a 2d-element creates 6 3d-elements, while the usual extrusion creates 4 in the high discretized grid and 2 in the low discretized grid
    nbc = mesh.nbc
    mesh3d.nel = nel3d
    mesh3d.ndim = 3
    # The curved sides will also be extruded, one on each side of each element along nz
    mesh3d.ncurv = 2*nz*mesh.ncurv

    # add extra copies of all elements
    for k in range(6*int(nz/4)-1):
        mesh3d.elem = mesh3d.elem + copy.deepcopy(mesh.elem)

    # set the x,y,z locations and curvature
    for k in range(0, nz, 4):
        for i in range(nel2d):
            iel = 6*(i + nel2d*int(k/4))
            for iell in range(6):
                mesh3d.elem[iel+iell] = copy.deepcopy(mesh.elem[i])
                mesh3d.elem[iel+iell].pos = np.zeros((3, 2, 2, 2))
            
            xvec = np.zeros((2,2))
            yvec = np.zeros((2,2))
            rvec = np.zeros((2,2))
            index_lo = np.zeros((2,2), dtype=int) #index of low points
            index_hi = np.zeros((2,2), dtype=int) #index of high points
            iindex_lo = 0
            iindex_hi = 0
            iedgelat = np.zeros((2), dtype=int)
            iedgeconlat = np.zeros((2), dtype=int)
            for ii in range(2):
                for jj in range(2):
                    xvec[jj,ii] = mesh.elem[i].pos[0,0,jj,ii]
                    yvec[jj,ii] = mesh.elem[i].pos[1,0,jj,ii]
                    rvec[jj,ii] = fun(xvec[jj,ii], yvec[jj,ii], Rlim)
                    if rvec[jj,ii] <= 0.0:
                        if (iindex_lo > 1):
                            logger.critical('Mid element not consistent. Criteria must divide elements with 2 points on each side.')
                            return -11
                        index_lo[iindex_lo,:]=[jj,ii]
                        iindex_lo += 1
                    else:
                        if (iindex_hi > 1):
                            logger.critical('Mid element not consistent. Criteria must divide elements with 2 points on each side.')
                            return -11
                        index_hi[iindex_hi,:]=[jj,ii]
                        iindex_hi += 1
            if (iindex_lo != 2) or (iindex_hi != 2):
                logger.critical('Mid element not consistent. Criteria must divide elements with 2 points on each side.')
                return -11
            
            # find the indices of edges, for curvature and boundary condition
            # iedgehi is the index of the edge of element iel+0 that is the intersection between iel+0 and iel+1 (high edge). iedgelo is the index of the edge of element iel+1 that is the intersection (low edge).
            # iedgelat are the indices of the lateral (splitted) edges. iedgeconlat are the indices of the edges (edge in z-direction) of elements iel+2 and iel+3 that connect to the respective lateral edges.
            if (index_lo[0,:]==[0,0]).all():
                if (index_hi[0,:]==[0,1]).all():
                    iedgehi = 1
                    iedgelo = 3
                    iedgelat[0] = 0
                    iedgelat[1] = 2
                    iedgeconlat[0] = 9
                    iedgeconlat[1] = 10
                else:
                    iedgehi = 2
                    iedgelo = 0
                    iedgelat[0] = 3
                    iedgelat[1] = 1
                    iedgeconlat[0] = 11
                    iedgeconlat[1] = 10
            elif (index_lo[0,:]==[1,0]).all():
                iedgehi = 0
                iedgelo = 2
                iedgelat[0] = 3
                iedgelat[1] = 1
                iedgeconlat[0] = 8
                iedgeconlat[1] = 9
            elif (index_lo[0,:]==[0,1]).all():
                iedgehi = 3
                iedgelo = 1
                iedgelat[0] = 0
                iedgelat[1] = 2
                iedgeconlat[0] = 8
                iedgeconlat[1] = 11
            
            # find x and y locations
            poslo = copy.deepcopy(mesh.elem[i].pos[:, :1, index_lo[:,0], index_lo[:,1]])
            poshi = copy.deepcopy(mesh.elem[i].pos[:, :1, index_hi[:,0], index_hi[:,1]])
            # mid position is influenced by curvature
            posmid  = 0.5*(mesh.elem[i].pos[:,:1, index_lo[:,0], index_lo[:,1]] + mesh.elem[i].pos[:,:1, index_hi[:,0], index_hi[:,1]])
            for ilat in range(2):
                #finds the mid points of lateral edges (also considering curvature)
                posmid[:,0,ilat] = mst.edge_mid(mesh.elem[i],iedgelat[ilat])
            
            # fill in the x and y location
            mesh3d.elem[iel].pos[:, 0:2, index_lo[:,0], index_lo[:,1]] = poslo
            mesh3d.elem[iel].pos[:, 0:2, index_hi[:,0], index_hi[:,1]] = posmid
            
            mesh3d.elem[iel+1].pos[:, 0:2, index_lo[:,0], index_lo[:,1]] = posmid
            mesh3d.elem[iel+1].pos[:, 0:2, index_hi[:,0], index_hi[:,1]] = poshi
            
            mesh3d.elem[iel+2].pos[:, 0:2, index_lo[:,0], index_lo[:,1]] = poslo
            mesh3d.elem[iel+2].pos[:,  :1, index_hi[:,0], index_hi[:,1]] = posmid
            mesh3d.elem[iel+2].pos[:, 1:2, index_hi[:,0], index_hi[:,1]] = poshi
            
            mesh3d.elem[iel+3].pos[:, 0:2, index_lo[:,0], index_lo[:,1]] = poslo
            mesh3d.elem[iel+3].pos[:,  :1, index_hi[:,0], index_hi[:,1]] = poshi
            mesh3d.elem[iel+3].pos[:, 1:2, index_hi[:,0], index_hi[:,1]] = posmid
            
            mesh3d.elem[iel+4].pos[:, 0:2, index_lo[:,0], index_lo[:,1]] = posmid
            mesh3d.elem[iel+4].pos[:, 0:2, index_hi[:,0], index_hi[:,1]] = poshi
            
            mesh3d.elem[iel+5].pos[:, 0:2, index_lo[:,0], index_lo[:,1]] = poslo
            mesh3d.elem[iel+5].pos[:, 0:2, index_hi[:,0], index_hi[:,1]] = posmid

            # fill in the z location
            mesh3d.elem[iel].pos[2, 0, :, :] = z1[k]
            mesh3d.elem[iel].pos[2, 1, :, :] = z2[k]
            
            mesh3d.elem[iel+1].pos[2, 0, :, :] = z1[k]
            mesh3d.elem[iel+1].pos[2, 1, index_lo[0,0], index_lo[0,1]] = z2[k]
            mesh3d.elem[iel+1].pos[2, 1, index_lo[1,0], index_lo[1,1]] = z2[k]
            mesh3d.elem[iel+1].pos[2, 1, index_hi[0,0], index_hi[0,1]] = z2[k+1]
            mesh3d.elem[iel+1].pos[2, 1, index_hi[1,0], index_hi[1,1]] = z2[k+1]
            
            mesh3d.elem[iel+2].pos[2, 0, :, :] = z1[k+1]
            mesh3d.elem[iel+2].pos[2, 1, :, :] = z2[k+1]
            
            mesh3d.elem[iel+3].pos[2, 0, :, :] = z1[k+2]
            mesh3d.elem[iel+3].pos[2, 1, :, :] = z2[k+2]
            
            mesh3d.elem[iel+4].pos[2, 1, :, :] = z2[k+3]
            mesh3d.elem[iel+4].pos[2, 0, index_lo[0,0], index_lo[0,1]] = z1[k+3]
            mesh3d.elem[iel+4].pos[2, 0, index_lo[1,0], index_lo[1,1]] = z1[k+3]
            mesh3d.elem[iel+4].pos[2, 0, index_hi[0,0], index_hi[0,1]] = z1[k+2]
            mesh3d.elem[iel+4].pos[2, 0, index_hi[1,0], index_hi[1,1]] = z1[k+2]
            
            mesh3d.elem[iel+5].pos[2, 0, :, :] = z1[k+3]
            mesh3d.elem[iel+5].pos[2, 1, :, :] = z2[k+3]
            
            for icurv in range(4):
                # Curvature type 's' acts on faces, not edges. Does not make sense for extruded mesh. Changing to 'C'.
                if mesh.elem[i].ccurv[icurv] == 's':
                    print('Curvature s on element', i+1,'. Not consistent with extrusion, changing to C')
                    mesh.elem[i].ccurv[icurv] == 'C'
                    mesh.elem[i].ccurv[icurv][0] = mesh.elem[i].ccurv[icurv][4]
                    mesh.elem[i].ccurv[icurv][1:4] = 0.0
                elif (mesh.elem[i].ccurv[icurv] != '') and (mesh.elem[i].ccurv[icurv] != 'm') and (mesh.elem[i].ccurv[icurv] != 'C'):
                    print('Warning: Curvature unknown on element', i+1,'. Unpredictable behaviour.')
            
            # extend curvature and correct it if necessary
            # calculate coordinates of midsize-node for every curvature type, except both empty (even if both are 'C'). Then find radius, if applicable. 'm' takes precendence over 'C'.
            #if mesh.elem[i].ccurv[iedgehi] != mesh.elem[i].ccurv[iedgelo]:
            if mesh.elem[i].ccurv[iedgehi] != '' or mesh.elem[i].ccurv[iedgelo] != '':
                midpointhi = mst.edge_mid(mesh.elem[i], iedgehi)
                midpointlo = mst.edge_mid(mesh.elem[i], iedgelo)
                midpointmid = 0.5*(posmid[:,0,0]+posmid[:,0,1]) + 0.5*(midpointhi-0.5*(poshi[:,0,0]+poshi[:,0,1])+midpointlo-0.5*(poslo[:,0,0]+poslo[:,0,1]))
                if mesh.elem[i].ccurv[iedgehi]=='m' or mesh.elem[i].ccurv[iedgelo]=='m':
                    mesh3d.elem[iel].ccurv[iedgehi] = 'm'
                    mesh3d.elem[iel+1].ccurv[iedgelo] = 'm'
                    mesh3d.elem[iel].curv[iedgehi][:3] = midpointmid
                    mesh3d.elem[iel+1].curv[iedgelo][:3] = midpointmid
                elif mesh.elem[i].ccurv[iedgehi]=='C' or mesh.elem[i].ccurv[iedgelo]=='C':
                    midpointmid[2] = z1[k]
                    curvmid = mst.edge_circle(mesh3d.elem[iel], iedgehi, midpointmid)
                    if curvmid[0] == 0.0:
                        mesh3d.elem[iel].ccurv[iedgehi] = ''
                        mesh3d.elem[iel+1].ccurv[iedgelo] = ''
                        mesh3d.elem[iel].curv[iedgehi][:4] = 0.0
                        mesh3d.elem[iel+1].curv[iedgelo][:4] = 0.0
                    else:
                        curvmid[1:4] = 0.0
                        mesh3d.elem[iel].ccurv[iedgehi] = 'C'
                        mesh3d.elem[iel+1].ccurv[iedgelo] = 'C'
                        mesh3d.elem[iel].curv[iedgehi][:4] = curvmid
                        mesh3d.elem[iel+1].curv[iedgelo][:4] = -curvmid
                else:
                    print('Warning: Splitted element curvature unknown on element', i+1,'of mid mesh. Removing curvature.')
                    #For cases not implemented, remove curvature
                    mesh3d.elem[iel].ccurv[iedgehi] = ''
                    mesh3d.elem[iel+1].ccurv[iedgelo] = ''
                    mesh3d.elem[iel].curv[iedgehi] = 0.0*(mesh.elem[i].curv[iedgehi])
                    mesh3d.elem[iel+1].curv[iedgelo] = copy.deepcopy(mesh3d.elem[iel].curv[iedgehi])
                
            for ilat in range(2):
                # Fixing curvature of edges divided in half. For curv_type == 'C', it is not a true extrusion - 'diagonal edges' not consistent with 'C'.
                if mesh.elem[i].ccurv[iedgelat[ilat]]=='m':
                    # coordinates of midsize-node is approximated. Considering an edge aligned with the x-axis, the position would be (ym_new = 3/4*ym_old = 1/2*ym_old(mean y-position)+1/4*ym_old(distance)) at xm_new=(x2-x1)/4. This comes from parabolic curve (however, it is not exactly midsize)
                    dmid = ((poshi[0,0,ilat]-poslo[0,0,ilat])*(poslo[1,0,ilat]-posmid[1,0,ilat])-(poslo[0,0,ilat]-posmid[0,0,ilat])*(poshi[1,0,ilat]-poslo[1,0,ilat]))/((poshi[0,0,ilat]-poslo[0,0,ilat])**2+(poshi[1,0,ilat]-poslo[1,0,ilat])**2)
                    mesh3d.elem[iel].curv[iedgelat[ilat]][0]   = 0.5*(poslo[0,0,ilat]+posmid[0,0,ilat]) + dmid/4.0*(poshi[1,0,ilat]-poslo[1,0,ilat])
                    mesh3d.elem[iel+1].curv[iedgelat[ilat]][0] = 0.5*(posmid[0,0,ilat]+poshi[0,0,ilat]) + dmid/4.0*(poshi[1,0,ilat]-poslo[1,0,ilat])
                    mesh3d.elem[iel].curv[iedgelat[ilat]][1]   = 0.5*(poslo[1,0,ilat]+posmid[1,0,ilat]) - dmid/4.0*(poshi[0,0,ilat]-poslo[0,0,ilat])
                    mesh3d.elem[iel+1].curv[iedgelat[ilat]][1] = 0.5*(posmid[1,0,ilat]+poshi[1,0,ilat]) - dmid/4.0*(poshi[0,0,ilat]-poslo[0,0,ilat])
                elif mesh.elem[i].ccurv[iedgelat[ilat]]=='C':
                    # if the lateral edge has curvature 'C', the diagonal edges connected to it (iedgeconlat of elements iel+2 and iel+3) would have curvature 'C', which are inconsistent with edges 8-12 inside Nek5000 (because these are supposed to be edges in z-direction). Changing to curvature 'm' for elements iel+1 (and iel+2, iel+3, iel+2, iel+4)
                    midpointlathi = mst.edge_mid(mesh3d.elem[iel+1], iedgelat[ilat])
                    mesh3d.elem[iel+1].curv[iedgelat[ilat]][:3] = midpointlathi
                    mesh3d.elem[iel+1].ccurv[iedgelat[ilat]] = 'm'
            
            for icurv in range(4):
                # a 2D element has 4 edges that can be curved, numbered 0-3;
                # the extruded 3D element can have four more (on the other side), numbered 4-7
                for iell in range(6):
                    mesh3d.elem[iel+iell].ccurv[icurv+4] = mesh3d.elem[iel+iell].ccurv[icurv]
                    mesh3d.elem[iel+iell].curv[icurv+4] = mesh3d.elem[iel+iell].curv[icurv]    # curvature params
                    
            mesh3d.elem[iel+2].curv[0:4] = copy.deepcopy(mesh3d.elem[iel].curv[0:4])
            mesh3d.elem[iel+3].curv[4:8] = copy.deepcopy(mesh3d.elem[iel].curv[0:4])
            mesh3d.elem[iel+5].curv      = copy.deepcopy(mesh3d.elem[iel].curv)
            mesh3d.elem[iel+4].curv      = copy.deepcopy(mesh3d.elem[iel+1].curv)
            mesh3d.elem[iel+2].ccurv[0:4] = copy.deepcopy(mesh3d.elem[iel].ccurv[0:4])
            mesh3d.elem[iel+3].ccurv[4:8] = copy.deepcopy(mesh3d.elem[iel].ccurv[0:4])
            mesh3d.elem[iel+5].ccurv      = copy.deepcopy(mesh3d.elem[iel].ccurv)
            mesh3d.elem[iel+4].ccurv      = copy.deepcopy(mesh3d.elem[iel+1].ccurv)
            
            for icurv in range(4):
                # z should be set to the proper value.
                curv_type = mesh3d.elem[iel].ccurv[icurv]
                if curv_type == 'm':
                    izcurv=2
                    mesh3d.elem[iel].curv[icurv][izcurv]     = z1[k]
                    mesh3d.elem[iel].curv[icurv+4][izcurv]   = z2[k]
                    mesh3d.elem[iel+2].curv[icurv][izcurv]   = z1[k+1]
                    mesh3d.elem[iel+2].curv[icurv+4][izcurv] = z2[k+1]
                    mesh3d.elem[iel+3].curv[icurv][izcurv]   = z1[k+2]
                    mesh3d.elem[iel+3].curv[icurv+4][izcurv] = z2[k+2]
                    mesh3d.elem[iel+5].curv[icurv][izcurv]   = z1[k+3]
                    mesh3d.elem[iel+5].curv[icurv+4][izcurv] = z2[k+3]
                
                curv_type = mesh3d.elem[iel+1].ccurv[icurv]
                # curvature of iel+1 may be different from iel because of diagonal edges
                if curv_type == 'm':
                    izcurv=2
                    mesh3d.elem[iel+1].curv[icurv][izcurv]   = z1[k]
                    mesh3d.elem[iel+4].curv[icurv+4][izcurv] = z2[k+3]
                    if icurv==iedgehi:
                        mesh3d.elem[iel+1].curv[iedgehi+4][izcurv] = z2[k+1]
                        mesh3d.elem[iel+4].curv[iedgehi][izcurv]   = z2[k+1]
                    elif icurv==iedgelo:
                        mesh3d.elem[iel+1].curv[iedgelo+4][izcurv] = z1[k+1]
                        mesh3d.elem[iel+4].curv[iedgelo][izcurv]   = z2[k+2]
                    elif icurv==iedgelat[0] or icurv==iedgelat[1]:
                        mesh3d.elem[iel+1].curv[icurv+4][izcurv]    = 0.5*(z1[k+1]+z2[k+1])
                        mesh3d.elem[iel+4].curv[icurv][izcurv]      = 0.5*(z2[k+1]+z2[k+2])
            
            #Fixing the curvature of 3d-edges in z-direction that connects to lateral edges in trapezoidal elements (all other edges in z-direction - indices 8 to 11 - do not have curvature)
            for ilat in range(2):
                mesh3d.elem[iel+2].curv[iedgeconlat[ilat]] = copy.deepcopy(mesh3d.elem[iel+1].curv[iedgelat[ilat]+4])
                mesh3d.elem[iel+2].ccurv[iedgeconlat[ilat]] = copy.deepcopy(mesh3d.elem[iel+1].ccurv[iedgelat[ilat]+4])
                mesh3d.elem[iel+3].curv[iedgeconlat[ilat]] = copy.deepcopy(mesh3d.elem[iel+4].curv[iedgelat[ilat]])
                mesh3d.elem[iel+3].ccurv[iedgeconlat[ilat]] = copy.deepcopy(mesh3d.elem[iel+4].ccurv[iedgelat[ilat]])
            
    # fix the internal boundary conditions (even though it's probably useless) 
    # the end boundary conditions will be overwritten later with the proper ones
            for ibc in range(nbc):
                # set the conditions in faces normal to z
                for iell in range(6):
                    mesh3d.elem[iel+iell].bcs[ibc, 4][0] = 'E'
                    mesh3d.elem[iel+iell].bcs[ibc, 4][1] = iel+iell+1
                    mesh3d.elem[iel+iell].bcs[ibc, 4][2] = 5
                    mesh3d.elem[iel+iell].bcs[ibc, 4][4] = 6
                    mesh3d.elem[iel+iell].bcs[ibc, 5][0] = 'E'
                    mesh3d.elem[iel+iell].bcs[ibc, 5][1] = iel+iell+1
                    mesh3d.elem[iel+iell].bcs[ibc, 5][2] = 6
                    mesh3d.elem[iel+iell].bcs[ibc, 5][4] = 5
                
                mesh3d.elem[iel].bcs[ibc, 4][3] = iel+1-nel2d
                mesh3d.elem[iel].bcs[ibc, 5][3] = iel+1+2
                mesh3d.elem[iel+1].bcs[ibc, 4][3] = iel+1-nel2d
                mesh3d.elem[iel+1].bcs[ibc, 5][3] = iel+1+2
                mesh3d.elem[iel+2].bcs[ibc, 4][3] = iel+1
                mesh3d.elem[iel+2].bcs[ibc, 5][3] = iel+1+3
                mesh3d.elem[iel+3].bcs[ibc, 4][3] = iel+1+2
                mesh3d.elem[iel+3].bcs[ibc, 5][3] = iel+1+5
                mesh3d.elem[iel+4].bcs[ibc, 4][3] = iel+1+3
                mesh3d.elem[iel+4].bcs[ibc, 5][3] = iel+1+nel2d
                mesh3d.elem[iel+5].bcs[ibc, 4][3] = iel+1+3
                mesh3d.elem[iel+5].bcs[ibc, 5][3] = iel+1+nel2d
                # update the conditions for side faces. (FIXME : Not corrected for merging - need to know numbering of other simply-extruded meshes (it is not really necessary, internal bc are not used))
                for iface in range(4):
                    for iell in range(6):
                        mesh3d.elem[iel+iell].bcs[ibc, iface][1] = iel+iell+1
                        mesh3d.elem[iel+iell].bcs[ibc, iface][2] = iface+1
                    if mesh3d.elem[iel].bcs[ibc, iface][0] == 'E':
                        # el.bcs[ibc, 0][1] ought to contain iel+1 once the mesh is valid
                        # but for now it should be off by a factor of nel2d because it is a copy of an element in the first slice
                        ielneigh = 6*(mesh3d.elem[iel].bcs[ibc, iface][3]-1 + nel2d*int(k/4))
                        for iell in range(6):
                            mesh3d.elem[iel+iell].bcs[ibc, iface][3] = ielneigh+1+iell
                
                # Correct internal bc for mid faces of elements.
                mesh3d.elem[iel].bcs[ibc, iedgehi][0] = 'E'
                mesh3d.elem[iel].bcs[ibc, iedgehi][3] = iel+1+1
                mesh3d.elem[iel].bcs[ibc, iedgehi][4] = iedgelo+1
                mesh3d.elem[iel+1].bcs[ibc, iedgelo][0] = 'E'
                mesh3d.elem[iel+1].bcs[ibc, iedgelo][3] = iel+1
                mesh3d.elem[iel+1].bcs[ibc, iedgelo][4] = iedgehi+1
                mesh3d.elem[iel+1].bcs[ibc, 5][0] = 'E'
                mesh3d.elem[iel+1].bcs[ibc, 5][3] = iel+1+2
                mesh3d.elem[iel+1].bcs[ibc, 5][4] = iedgehi+1
                mesh3d.elem[iel+2].bcs[ibc, iedgehi][0] = 'E'
                mesh3d.elem[iel+2].bcs[ibc, iedgehi][3] = iel+1+1
                mesh3d.elem[iel+2].bcs[ibc, iedgehi][4] = 6
                mesh3d.elem[iel+3].bcs[ibc, iedgehi][0] = 'E'
                mesh3d.elem[iel+3].bcs[ibc, iedgehi][3] = iel+1+4
                mesh3d.elem[iel+3].bcs[ibc, iedgehi][4] = 5
                mesh3d.elem[iel+4].bcs[ibc, 4][0] = 'E'
                mesh3d.elem[iel+4].bcs[ibc, 4][3] = iel+1+3
                mesh3d.elem[iel+4].bcs[ibc, 4][4] = iedgehi+1
                mesh3d.elem[iel+4].bcs[ibc, iedgelo][0] = 'E'
                mesh3d.elem[iel+4].bcs[ibc, iedgelo][3] = iel+1+5
                mesh3d.elem[iel+4].bcs[ibc, iedgelo][4] = iedgehi+1
                mesh3d.elem[iel+5].bcs[ibc, iedgehi][0] = 'E'
                mesh3d.elem[iel+5].bcs[ibc, iedgehi][3] = iel+1+4
                mesh3d.elem[iel+5].bcs[ibc, iedgehi][4] = iedgelo+1
                
    # now fix the end boundary conditions 
    # face 5 is at zmin and face 6 is at zmax (with Nek indexing, corresponding to 4 and 5 in Python)
    for i in range(0,6*nel2d,6):
        for ibc in range(nbc):
            i1 = i+nel3d-6*nel2d+5
            mesh3d.elem[i].bcs[ibc, 4][0] = bc1
            mesh3d.elem[i].bcs[ibc, 4][1] = i+1
            mesh3d.elem[i].bcs[ibc, 4][2] = 5
            mesh3d.elem[i+1].bcs[ibc, 4][0] = bc1
            mesh3d.elem[i+1].bcs[ibc, 4][1] = i+1+1
            mesh3d.elem[i+1].bcs[ibc, 4][2] = 5
            mesh3d.elem[i1].bcs[ibc, 5][0] = bc2
            mesh3d.elem[i1].bcs[ibc, 5][1] = i1+1
            mesh3d.elem[i1].bcs[ibc, 5][2] = 6
            mesh3d.elem[i1-1].bcs[ibc, 5][0] = bc2
            mesh3d.elem[i1-1].bcs[ibc, 5][1] = i1-1+1
            mesh3d.elem[i1-1].bcs[ibc, 5][2] = 6
            
            mesh3d.elem[i].bcs[ibc, 4][3] = 0.0
            mesh3d.elem[i].bcs[ibc, 4][4] = 0.0
            mesh3d.elem[i+1].bcs[ibc, 4][3] = 0.0
            mesh3d.elem[i+1].bcs[ibc, 4][4] = 0.0
            mesh3d.elem[i1].bcs[ibc, 5][3] = 0.0
            mesh3d.elem[i1].bcs[ibc, 5][4] = 0.0
            mesh3d.elem[i1-1].bcs[ibc, 5][3] = 0.0
            mesh3d.elem[i1-1].bcs[ibc, 5][4] = 0.0
            
            # fix the matching faces for the periodic conditions
            if bc1 == 'P':
                mesh3d.elem[i].bcs[ibc, 4][3] = i1+1
                mesh3d.elem[i].bcs[ibc, 4][4] = 6
                mesh3d.elem[i+1].bcs[ibc, 4][3] = i1-1+1
                mesh3d.elem[i+1].bcs[ibc, 4][4] = 6
            if bc2 == 'P':
                mesh3d.elem[i1].bcs[ibc, 5][3] = i+1
                mesh3d.elem[i1].bcs[ibc, 5][4] = 5
                mesh3d.elem[i1-1].bcs[ibc, 5][3] = i+1+1
                mesh3d.elem[i1-1].bcs[ibc, 5][4] = 5
    
    # Removing internal boundary conditions. 'E' boundary conditions are wrong where meshes merge, but should be right internally. (When possible: FIXME indices and delete these lines. However, it is not really necessary, internal bc are not used)
    for (iel, el) in enumerate(mesh3d.elem):
        for ibc in range(mesh3d.nbc):
            for iface in range(6):
                el.bcs[ibc, iface][1] = iel+1
                if el.bcs[ibc, iface][0] == 'E':
                    el.bcs[ibc, iface][0] = ''
                    el.bcs[ibc, iface][1] = iel+1
                    el.bcs[ibc, iface][2] = iface+1
                    el.bcs[ibc, iface][3] = 0.0
                    el.bcs[ibc, iface][4] = 0.0
    
    # FIND THE CURVED ELEMENTS
    ncurv = 0
    for el in mesh3d.elem:
        for iedge in range(12):
            if el.ccurv[iedge] != '':
                ncurv = ncurv+1
    mesh3d.ncurv = ncurv
    
    # return the extruded mesh
    return mesh3d

def merge(mesh, other):
    """
    Merges another exadata into the current one. Does not connect it, to save time
    Parameters
    ----------
    other: exadata
           mesh to merge into mesh
    """
    # perform some consistency checks
    if mesh.ndim != other.ndim:
        logger.error(f'Cannot merge meshes of dimensions {mesh.ndim} and {other.ndim}!')
        return -1
    if mesh.lr1[0] != other.lr1[0]:
        logger.error('Cannot merge meshes of different polynomial orders ({} != {})'.format(mesh.lr1[0], other.lr1[0]))
        return -2

    # add the new elements (in an inconsistent state if there are internal boundary conditions)
    nel1 = mesh.nel
    mesh.nel = mesh.nel + other.nel
    mesh.ncurv = mesh.ncurv + other.ncurv
    mesh.elmap = np.concatenate((mesh.elmap, other.elmap+nel1))
    # the deep copy is required here to avoid leaving the 'other' mesh in an inconsistent state by modifying its elements
    mesh.elem = mesh.elem + copy.deepcopy(other.elem)
    
    # check how many boundary condition fields we have
    nbc = min(mesh.nbc, other.nbc)
    
    # correct the boundary condition numbers:
    # the index of the elements and neighbours have changed
    for iel in range(nel1, mesh.nel):
        for ibc in range(other.nbc):
            for iface in range(6):
                mesh.elem[iel].bcs[ibc, iface][1] = iel+1
                bc = mesh.elem[iel].bcs[ibc, iface][0]
                if bc == 'E' or bc == 'P':
                    neighbour = mesh.elem[iel].bcs[ibc, iface][3]
                    mesh.elem[iel].bcs[ibc, iface][3] = neighbour + nel1
    
#   for (iel, el) in enumerate(mesh.elem):
#       for ibc in range(mesh.nbc):
#           for iface in range(6):
#               el.bcs[ibc, iface][1] = iel+1
    
    return mesh
