import numpy as np
import matplotlib.pyplot as plt
from model import BeamModel
import postprocess
import utilities

'''
coordinate system: x -> longitudinal axis, y -> perpendicular
'''

parameters = {
                'n_elements': 10,
                'lx_total_beam': 10.0,
                'shear_only': False, # see bernoulli_element
                'decouple': True,
                'dofs_per_node': 2, #y and g  
                'dof_labels': ['y','g'],
                'material_density': 160, 
                'E_Modul': 1,#2.861e8,
                'nu': 1.0, #3D only
                'nodes_per_elem': 2,
                'cross_section_area': 1.0,
                'Iy': 10000.0,
                'Iz': 1.0, #3D only
                'It': 1.0,
                'bc': [0,1], # ids of dofs that are fixed
                'modes_to_consider': 15,
                'static_load_magnitude': -1.0,
                
}

# NOTE: if optimize decouple and/or shear_only must be true
optimize = True

# NOTE: targets static_disp, frequency & eigenform need: shear_only = False, decouple = True 
# targets frequency_shear needs: shear_only and decouple = True
targets = ['static_tip_disp', 'frequency','frequency_shear', 'eigenform']
opt_methods = ['Nelder-Mead', 'Powell', 'BFGS', 'L-BFGS-B','TNC', 'CG', 'SLSQP']

method =  opt_methods[0]

target = targets[3]
'''
actual values of k_yg:
3 elements: 5 400
10 elements: 60 000
'''
if optimize:
    optimization_parameters = {
                                'optimization_target':target,
                                'bounds': (-10000,10000), # 
                                'init_guess_1': 10000,
                                'init_guess_2': (50000, 70000), # array of 2 entries if 2 variables are optimized
                                'plot_opt_func': False, # evaluated between the 'bounds'
                                'method': method,
                                'use_minimize_scalar': False,
                                'scaling_exponent': 2, # scaling of objective function result value
                                'modes_to_consider': 3
    }

else: 
    optimization_parameters = None
    
# create the beam model
beam = BeamModel(parameters, optimization_parameters)

# call the analysis/ could be incorporated in the model itself
beam.static_analysis_solve()
beam.eigenvalue_solve()

# plot optimization parameters
if optimize:
    postprocess.plot_optimization_parameters(beam)
    postprocess.plot_multiple_result_vectors(beam, beam.disp_vector)


# POSTPROCESS THE RESULTS 
postprocess.plot_static_result(beam)
postprocess.plot_eigenmodes(beam, fit = False, analytic=True)

# OBJECTIVE FUNCTION: plot is here at the and since it changes properties of the beam directly
if optimize: 
    if optimization_parameters['plot_opt_func']:
        if optimization_parameters['optimization_target'] == 'frequency_shear':
            postprocess.plot_objective_function_3D(beam)
        else:
            postprocess.plot_objective_function_2D(beam)
