import numpy as np
import pymech.neksuite as ns
import pymech.exadata as exa
import meshtools as mst

path = '../examples/'
meshI = 'in_mesh_2d.rea'
meshO = 'out_mesh_3d'
#meshI = 'boxmix.rea'

## defining paths to files
fnameI = path + meshI
fnameO = path + meshO

# Reading 2D mesh
if fnameI[-3:]=='rea':
    mesh2D = ns.readrea(fnameI)
elif fnameI[-3:]=='re2':
    mesh2D = ns.readre2(fnameI)
else:
    print('Assuming mesh has .rea format')
    mesh2D = ns.readrea(fnameI+'.rea')

#Input parameters
z = [-1.0,1.0]
n = 32
bc1='v'
bc2='O'

imesh_high=1 #index of mesh with higher discretization. Example: for imesh_high=0, the mesh with higher discretization is the most internal mesh (for for imesh_high=1, it is the second most internal mesh)
funpar=[0.53, 1.25, 2.0, 3.6]

R0=[0.53, 1.25, 2.0, 3.6]
for ifun in range(4):
    splitpoint = [R0[ifun],0.01,0.0]
    splitpointneig = [R0[ifun],-0.01,0.0]
    ielsplit = mst.iel_point(mesh2D, splitpoint)
    ielsplitneig = mst.iel_point(mesh2D, splitpointneig)
    iedge0 = mst.iface_neig(mesh2D, ielsplit, ielsplitneig)
    xyzline = mst.lim_polyg(mesh2D, ielsplit, iedge0)
    funpar[ifun] = xyzline

fun=[mst.fun_polyg, mst.fun_polyg, mst.fun_polyg, mst.fun_polyg]
zlist = mst.define_z(z,n)
mesh3D = mst.extrude_split(mesh2D, zlist, bc1, bc2, fun, funpar, imesh_high)

##Input parameters 2
#z = [-1.0,1.0]
#n = 4
#bc1='v'
#bc2='O'
#imesh_high=0
#funpar=[0.01]
#fun_line = lambda xpos, ypos, rlim: ypos/rlim - 1.0
#fun=[fun_line]
#
#zlist = mst.define_z([-1.0,1.0,0.03],16,'gpdzn')
#mesh3D = mst.extrude_split(mesh2D, zlist, bc1, bc2, fun, funpar, imesh_high)
#
#z0=z[0]
#ble = 10.0 # sweep angle (degrees) at the leading edge (positive is converging for increasing z)
#bte = 20.0 # sweep angle (degrees) at the trailing edge (positive is converging for increasing z)
#dih = 10.0 # dihedral angle (degrees) at the trailing edge
#mesh3D = mst.taper(mesh3D, ble, bte, dih)

ns.writerea(fnameO+'.rea',mesh3D)
ns.writere2(fnameO+'.re2',mesh3D)

print('End of the program')
