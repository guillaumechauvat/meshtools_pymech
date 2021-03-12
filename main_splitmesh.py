import pymech.neksuite as ns
import pymech.exadata as exa
import meshtools_split as msts
import splitmesh_fun as smf

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
n = [32]
bc1='v'
bc2='O'
coord = [''] # x and y positions of the leading and trailing edge
ble = 0.0 # sweep angle (degrees) at the leading edge (positive is converging for increasing z)
bte = 0.0 # sweep angle (degrees) at the trailing edge (positive is converging for increasing z)

imesh_high=1 #index of mesh with higher discretization. Example: for imesh_high=0, the mesh with higher discretization is the most internal mesh (for for imesh_high=1, it is the second most internal mesh)
funpar=[0.53, 1.25, 2.0, 3.6]

#fun_circ = lambda xpos, ypos, rlim: ((xpos**2+ypos**2)**0.5)/rlim - 1.0
#fun=[smf.fun_hexag, fun_circ, fun_circ, fun_circ]


R0=[0.53, 1.25, 2.0, 3.6]
for ifun in range(4):
    splitpoint = [R0[ifun],0.01,0.0]
    splitpointneig = [R0[ifun],-0.01,0.0]
    ielsplit = smf.iel_point(mesh2D, splitpoint)
    ielsplitneig = smf.iel_point(mesh2D, splitpointneig)
    iedge0 = smf.iface_neig(mesh2D, ielsplit, ielsplitneig)
    xyzline = smf.lim_polyg(mesh2D, ielsplit, iedge0)
    funpar[ifun] = xyzline

fun=[smf.fun_polyg, smf.fun_polyg, smf.fun_polyg, smf.fun_polyg]

#n = [4]
#imesh_high=0
#funpar=[0.01]
#fun_line = lambda xpos, ypos, rlim: ypos/rlim - 1.0
#fun=[fun_line]

mesh3D = msts.extrude(mesh2D, z, n, bc1, bc2, fun, funpar, imesh_high)
#mesh3D = msts.extrude(mesh2D, z, n, bc1, bc2)



#z0=z[0]
#ble = 10.0
#bte = 20.0
#dih = 10.0
#mesh3D = msts.taper(mesh3D, ble, bte, dih, coord, z0)
#mesh3D = msts.taper(mesh3D, ble, bte, dih)
#mesh3D = msts.extrude_taper(mesh2D, z, n, bc1, bc2, coord, ble, bte)

ns.writerea(fnameO+'.rea',mesh3D)
ns.writere2(fnameO+'.re2',mesh3D)

print('End of the program')
