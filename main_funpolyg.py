import pymech.neksuite as ns
import pymech.exadata as exa
import numpy as np
import meshtools_split as msts
import splitmesh_fun as smf

mesh2D = ns.readrea('../examples/in_mesh_2dfkd.rea')
#mesh2D = ns.readrea('../examples/in_mesh_2d.rea')
xyzpoint = [0.1,3.1,0.0]
iel = smf.iel_point(mesh2D, xyzpoint)
iedge0 = 2 #still need to define function that finds iedge
xyzline = smf.lim_polyg(mesh2D, iel, iedge0)
print(xyzline)

funval = smf.fun_polyg(-1.0, -3.0, xyzline)
print(funval)
