import pymech.neksuite as ns
import pymech.exadata as exa
import meshtools_split as msts
import meshtools8 as mst
import splitmesh_fun as smf

path = './'
meshI = 'in_mesh_2d.rea'
meshO = 'out_mesh_3d'

#meshI = 'rot2.rea'
#meshO = 'rot2_3d'

z = [-1.0,1.0]
n = [32]
bc1='v'
bc2='O'
coord = [''] # x and y positions of the leading and trailing edge
ble = 0.0 # sweep angle (degrees) at the leading edge (positive is converging for increasing z)
bte = 0.0 # sweep angle (degrees) at the trailing edge (positive is converging for increasing z)

imesh_high=1 #index of mesh with higher discretization. Example: for imesh_high=0, the mesh with higher discretization is the most internal mesh (for for imesh_high=1, it is the second most internal mesh)
Rlim=[0.53, 1.25, 2.0, 3.6]

fun_circ = lambda xpos, ypos, rlim: ((xpos**2+ypos**2)**0.5)/rlim - 1.0
fun=[smf.fun_hexag, fun_circ, fun_circ, fun_circ]

meshI = 'boxmix.rea'
n = [4]
imesh_high=0
Rlim=[0.01]
fun_line = lambda xpos, ypos, rlim: ypos/rlim - 1.0
fun=[fun_line]

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

mesh3D = msts.extrude_split(mesh2D, z, n, bc1, bc2, fun, Rlim, imesh_high)
#mesh3D = msts.extrude_split(mesh2D, z, n, bc1, bc2)
#mesh3D = mst.extrude(mesh2D, z, n, bc1, bc2)



#z0=z[0]
#ble = 10.0
#bte = 10.0
#dih = 0.0
#mesh3D = mst.taper(mesh3D, ble, bte, dih, coord, z0)
#mesh3D = mst.taper(mesh3D, ble, bte, dih)
#mesh3D = mst.extrude(mesh2D, z, n, bc1, bc2, coord, ble, bte)

ns.writerea(fnameO+'.rea',mesh3D)
ns.writere2(fnameO+'.re2',mesh3D)

print('End of the program')
