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
                'n_elements': 10,
                'lx_total_beam': 240,#180.0, #
                'material_density': 160,#150, 
                'E_Modul': 286100000.0,#8750.0,
                'nu': 0.1, #3D only
                'nodes_per_elem': 2,
                'cross_section_area': 72*24,#1350, #
                'Iy': 746496.0,#105241.5*g1,#1650.0,
                'Iz': 82944.0,#141085.0*g2,#1750.0, #3D only # associated with stiffnes K_y : bending around z displacement in y
                'I_param':100000.0, # used for the coupling eintries to make it independet of the others
                'It': 829440.0,#270281.1,#3400.0,
                'modes_to_consider': 15,
                'static_load_magnitude': -20.0,
                'inital_params_yg': [1.0,1.0,1.0],#[0.001,0.0012,0.0014]
                'inital_params_k_ya': [0,0],#[1.0, 1.0, 1.0, 1.0] # omega: stiffness coupling y - a omega1: y -g
                'inital_params_m_ya': [0.0,0.0,0.0]#[1.0, 1.0, 1.0] # omega: stiffness coupling, psi1, psi2: mass y-a, psi3: mass g-a
}


opt_methods = ['Nelder-Mead', 'Powell', 'BFGS', 'L-BFGS-B','TNC', 'CG', 'SLSQP']


ws = [[0.33,0.33,.33],[.4,.4,.2],[.45,.45,.1],[.8,.1,.1],[.1,.8,.1],[.2,.4,.4],[0.1,0.1,0.8],[0.5,0.5,0.]]

optimization_parameters = {
                            'caarc_freqs': [0.231,0.429,0.536], 
                            'consider_mode':0,
                            'method': opt_methods[-1],
                            'init_guess':[0.,0., 0.],#[0.1, 3]
                            'bounds':((0.0001, 100),(0.01, 100),(.0000001, 100)), #NOTE m seems to be corrected down always 
                            'weights':ws[-1]
                            }

# CREATE AN INITAL BEAM
beam = BeamModel(parameters, optimize_frequencies_init=True)
opt_geometric_props = beam.get_optimized_geometric_params()

postprocess.plot_eigenmodes_3D(beam, number_of_modes=3, dofs_to_plot=['y','a','z'], max_normed=False, fig_title='after init freq adjustments')

# CREATE AN OPTIMIZATION
coupling_opt = Optimizations(beam, optimization_parameters)

#postprocess.plot_eigenmodes_3D(beam, number_of_modes=3, dofs_to_plot=['y','a','z'], max_normed=False)

#postprocess.plot_objective_function_2D(coupling_opt, 'Iz')

# ==========================
# BENDING OPTIMIZATIONS
#coupling_opt.eigen_scalar_opt_yg()
#coupling_opt.eigen_vectorial_opt()

# ==========================
# TORSION OPTIMIZATIONS
#coupling_opt.eigen_ya_stiffness_opt()
coupling_opt.eigen_vectorial_k_ya_opt()
opt_params = coupling_opt.optimized_design_params

postprocess.plot_eigenmodes_3D(beam, 
                               opt_targets = coupling_opt.targets, 
                               initial=coupling_opt.inital, 
                               dofs_to_plot=['y','z','a'],
                               max_normed=False, 
                               do_rad_scale=False,
                               number_of_modes =3,
                               opt_params=optimization_parameters,
                               fig_title='after coupling opt')

# ===========================
# READJUSTING FREQUENCIES
# have to set all the optimized parameters 
parameters['Iy'] = opt_geometric_props['Iy'][0]
parameters['Iz'] = opt_geometric_props['Iz'][0]
parameters['It'] = opt_geometric_props['It'][0]
parameters['inital_params_k_ya'] = [opt_params[0],opt_params[1]]
parameters['inital_params_m_ya'] = [opt_params[2],opt_params[2],0.0]

coupled_beam = BeamModel(parameters, optimize_frequencies_init=False)

postprocess.plot_eigenmodes_3D(coupled_beam, number_of_modes=3, dofs_to_plot=['y','a','z'], max_normed=False, fig_title='reinitialized coupled beam')


print ('\nOPTIMIZATION OF FREQUENCY AFTER COUPLING')
freq_opt = Optimizations(model=coupled_beam)
freq_opt.adjust_sway_z_stiffness_for_target_eigenfreq(optimization_parameters['caarc_freqs'][0], 
                                                          target_mode = 0,
                                                          print_to_console=False)
# freq_opt.adjust_torsional_stiffness_for_target_eigenfreq(optimization_parameters['caarc_freqs'][2], 
#                                                           target_mode = 2,
#                                                           print_to_console=False)

print ('\ncomp_k alpha row after all opts:')
print (coupled_beam.comp_k[3])
# ===============================
# PLOTS AFTER OPTIMIZATIONS
postprocess.plot_eigenmodes_3D(coupled_beam, 
                               opt_targets = coupling_opt.targets, 
                               initial=coupling_opt.inital, 
                               dofs_to_plot=['y','a'],
                               max_normed=False, 
                               do_rad_scale=True,
                               number_of_modes =3,
                               opt_params=optimization_parameters,
                               fig_title='after coupling and readjustment of freqeuncies')

#postprocess.plot_objective_function_2D(optimization, design_var_label='k_ya')
#postprocess.plot_objective_function_2D(optimization)

# =============================
# OBJECTIVE FUNCTION PLOTS MUST BE DONE AFTER ACTUAL OPTIMIZATION
#postprocess.plot_objective_function_3D(optimization)

