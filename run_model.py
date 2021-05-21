import numpy as np
import matplotlib.pyplot as plt
from model import BeamModel
from optimizations import Optimizations
import postprocess
import utilities

'''
coordinate system: x -> longitudinal axis, y -> perpendicular
'''
g1 = 1.0
g2 = 1.0

parameters = {  'dimension': '3D',
                'n_elements': 3,
                'lx_total_beam': 180.0,
                'material_density': 160,#150, 
                'E_Modul': 286100000.0,#8750.0,
                'nu': 0.1, #3D only
                'nodes_per_elem': 2,
                'cross_section_area': 1350,
                'Iy': 105241.5*g1,#1650.0,
                'Iz': 141085.0*g2,#1750.0, #3D only
                'It': 270281.1,#3400.0,
                'modes_to_consider': 15,
                'static_load_magnitude': -20.0,
                'inital_params_yg': [1.0,1.0,1.0],#[0.001,0.0012,0.0014]
                'inital_params_k_ya': [0,0],#[1.0, 1.0, 1.0, 1.0] # omega: stiffness coupling y - a omega1: y -g
                'inital_params_m_ya': [0.0,0.0,0.0]#[1.0, 1.0, 1.0] # omega: stiffness coupling, psi1, psi2: mass y-a, psi3: mass g-a
}


opt_methods = ['Nelder-Mead', 'Powell', 'BFGS', 'L-BFGS-B','TNC', 'CG', 'SLSQP']


ws = [[0.33,0.33,.33],[.4,.4,.2],[.8,.1,.1],[0.1,0.1,0.8]]

optimization_parameters = {'consider_mode':0,
                            'method': opt_methods[-1],
                            'init_guess':[0,0],#[0.1, 3]
                            'weights':ws[0]}
# create the beam model
beam = BeamModel(parameters)#, optimization_parameters)
# an inital solve needed for optimization if it takes the original values to make a target
beam.eigenvalue_solve()
#postprocess.plot_eigenmodes_3D(beam, number_of_modes=1, dofs_to_plot=['y','a'], max_normed=False)

# CREATE AN OPTIMIZATION
optimization = Optimizations(beam, optimization_parameters)

# ==========================
# BENDING OPTIMIZATIONS
#optimization.eigen_scalar_opt_yg()
#optimization.eigen_vectorial_opt()

# ==========================
# TORSION OPTIMIZATIONS
#optimization.eigen_ya_stiffness_opt()
optimization.eigen_vectorial_k_ya_opt()

# =============================
# TESTING VALUES FOR THE MASS MATRIX
# vals = np.arange(-1,100,0.05)

# for val in vals:
#     m_params = [val,val,val]
#     beam.build_system_matricies(params_m_ya=m_params)
#     beam.eigenvalue_solve()

#print ('\nparameters working:\n', beam.working_params)

# ===============================
# PLOTS AFTER OPTIMIZATIONS
postprocess.plot_eigenmodes_3D(beam, 
                               opt_targets = optimization.targets, 
                               initial=optimization.inital, 
                               dofs_to_plot=['y','a'],
                               max_normed=False, 
                               do_rad_scale=False,
                               opt_params=optimization_parameters)

#postprocess.plot_objective_function_2D(optimization, design_var_label='k_ya')
#postprocess.plot_objective_function_2D(optimization)

# =============================
# OBJECTIVE FUNCTION PLOTS MUST BE DONE AFTER ACTUAL OPTIMIZATION
postprocess.plot_objective_function_3D(optimization)

