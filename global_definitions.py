

dof_lables = {'2D': ['y','g'], '3D':['x', 'y', 'z', 'a', 'b', 'g']}
n_dofs_node = {'2D':2, '3D':6}
dofs_of_bc = {'2D': [0,1], '3D':[0,1,2,3,4,5] }
tip_load = {'2D': -2, '3D':-5}

RESPONSE_DIRECTION_MAP = {'Qx':'x', 'Qy':'y', 'Qz':'z', 'Mx':'a', 'My':'b', 'Mz':'g'}

DIRECTION_LOAD_MAP = {'x':'Fx', 'y':'Fy', 'z':'Fz', 'a':'Mx', 'b':'My', 'g':'Mz'}

direction_response_map = {'x':'Qx', 'y':'Qy', 'z':'Qz', 'a':'Mx', 'b':'My', 'g':'Mz'}

greek = {'y':'y','z':'z', 'x':'x','a':r'\alpha', 'b':r'\beta', 'g':r'\gamma'}
