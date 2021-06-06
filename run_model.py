import numpy as np
import matplotlib.pyplot as plt
from model import BeamModel
from optimizations import Optimizations
from postprocess import Postprocess
import utilities
from plot_settings import plot_settings
from dynamic_analysis import DynamicAnalysis
from inputs import model_parameters
'''
coordinate system: x -> longitudinal axis, y -> perpendicular
'''
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]

width = utilities.cm2inch(7.3)
height = utilities.cm2inch(10)

params = plot_settings.get_params(w =width, h=height)

plt.rcParams.update({'axes.formatter.limits':(-3,3)}) 
#plt.rcParams.update({'figure.figsize': (width, height)})
#plt.rcParams.update(params)

# # INITIALIZING A POSTPROCESSING OBJECT
postprocess = Postprocess(show_plots = True, savefig = False, savefig_latex = False)

if postprocess.savefig or postprocess.savefig_latex:
   print ('\FIGURE SAVINGS ACTIVE')

run_options = {
                  'plot_inital':True, # with ajdusted frequencies
                  'plot_iteration':False,
                  'plot_coupled':False,
                  'plot_coupled_readjusted':True,
                  'plot_obj_func':False, 
                  'dynamic_analysis':False,
                  'static_analysis':False
                 }

# # MODEL GEOMETRIC, MATERIAL AND ELEMENT PARAMETERS
parameters = {  
                'dimension': '3D',
                'n_elements': 3,
                'lx_total_beam': 240,#180.0, #
                'material_density': 160,#150, 
                'E_Modul': 286100000.0,#8750.0,
                'nu': 0.1, #3D only
                'damping_coeff': 0.025,
                'nodes_per_elem': 2,
                'cross_section_area': 72*24,#1350, 
                'Iy': 746496.0,#105241.5,#1650.0,
                'Iz': 82944.0,#141085.0,#1750.0, 
                'I_param':100000.0, # used for the coupling eintries to make it independet of the others -> initialy in the scale of Iz or Iy
                'It': 829440.0,#270281.1,#3400.0,
                'modes_to_consider': 15,
                'static_load_magnitude': -20.0,
                'dynamic_load_file': "inputs\\dynamic_force_4_nodes.npy",
                'inital_params_yg': [1.0,1.0,1.0],#[0.001,0.0012,0.0014]
                'params_k_ya': [0,0],#[1.0, 1.0, 1.0, 1.0] # omega: stiffness coupling y - a omega1: y -g
                'params_m_ya': [0.0,0.0,0.0],#[1.0, 1.0, 1.0] # omega: stiffness coupling, psi1, psi2: mass y-a, psi3: mass g-a
                'eigen_freqs': [0.231,0.429,0.536]
             }

parameters = model_parameters.parameters_A

# # AVAILABLE OPTIMIZATION OPTIONS AND INPUT PARAMETER SETTINGS
opt_methods = ['Nelder-Mead', 'SLSQP', 'Powell', 'BFGS', 'L-BFGS-B','TNC', 'CG']

optimization_variables = ['kya','kga','both']

weights = [[0.33,0.33,.33],[0.5,0.5,0.],[.4,.4,.2],[.45,.45,.1],[.8,.1,.1],[.1,.8,.1],[.2,.4,.4],[0.1,0.1,0.8]]

optimization_parameters_A = {
                            'caarc_freqs_A': [0.231,0.429,0.536], 
                            'caarc_freqs_B': [0.2,0.23,0.4], 
                            'consider_mode':0,
                            'var_to_optimize' :optimization_variables[2],#'ya'#'ga' both
                            'include_mass':False,
                            'method': opt_methods[1],
                            'init_guess':[0.,10.,0],#[-2.,5., 0.],#[0.1, 3] # k_ya, k_ga, m_cross
                            'bounds':((0.01, 100),(0.01, 100),(.0000001, 100)), #NOTE m seems to be corrected down always 
                            'weights':weights[1],
                            'save_optimized_parameters':False
                            }

optimization_parameters_B = {
                            'caarc_freqs_A': [0.231,0.429,0.536], 
                            'caarc_freqs_B': [0.2,0.23,0.4], 
                            'consider_mode':0,
                            'var_to_optimize' :optimization_variables[2],#'ya'#'ga' both
                            'include_mass':False,
                            'method': opt_methods[1],
                            'init_guess':[0.,10.,0],#[-2.,5., 0.],#[0.1, 3] # k_ya, k_ga, m_cross
                            'bounds':((0.01, 100),(0.01, 100),(.0000001, 100)), #NOTE m seems to be corrected down always 
                            'weights':weights[1],
                            'save_optimized_parameters':False
                            }

optimization_parameters = optimization_parameters_A

# # DYNAMIC ANALYSIS
analysis_parameters = { "settings": 
                           {
                           "solver_type": "Linear",
                           "run_in_modal_coordinates": False,
                           "time":{
                                    "integration_scheme": "GenAlpha",
                                    "start": 0.0,
                                    "end": 600.0,
                                    "step" : 0.02},
                           "intial_conditions": {
                                    "displacement": None,
                                    "velocity": None,
                                    "acceleration" : None}
                           },
                        "input": 
                           {
                           "file_path": "inputs\\dynamic_force_4_nodes.npy"
                           }
                        }
               
# CREATE AN INITAL BEAM
beam = BeamModel(parameters, optimize_frequencies_init=True, use_translate_matrix=False)

if run_options['dynamic_analysis']:
   dynamic_analysis = DynamicAnalysis(beam, parameters = analysis_parameters)
   dynamic_analysis.solve()
   postprocess.plot_dynamic_results(dynamic_analysis, dof_label = 'g', node_id = 3, result_variable = 'reaction')
   postprocess.plot_fft(dof_label = 'g', dynamic_analysis = dynamic_analysis )

if run_options['static_analysis']:
   beam.static_analysis_solve(apply_mean_dynamic=True, direction='all')
   static_deformation_init = beam.static_deformation.copy()
   postprocess.plot_static_result(beam, load_type='mean', dofs_to_plot=['y','a'], do_rad_scale=True, save_suffix='init')

if run_options['plot_inital']:
   postprocess.plot_eigenmodes_3D(beam, 
                                 number_of_modes = 3, 
                                 dofs_to_plot=['y','a','z'], 
                                 max_normed=False, 
                                 plot_weights_in_title = False,
                                 fig_title= r'after initial frequency adjustments',
                                 filename_for_save = 'after initial freuqency opts')
   

# CREATE AN OPTIMIZATION
coupling_opt = Optimizations(beam, optimization_parameters)

# ==========================
# BENDING OPTIMIZATIONS
#coupling_opt.eigen_scalar_opt_yg()
#coupling_opt.eigen_vectorial_opt()

# ==========================
# TORSION COUPLING OPTIMIZATIONS
var_to_optimize = optimization_parameters['var_to_optimize']
if var_to_optimize != 'both':
   # optimize onyl one stiffness variable
   coupling_opt.eigen_ya_stiffness_opt(which = var_to_optimize)   
else:
   # optimizing both stiffness variabels
   coupling_opt.eigen_vectorial_ya_opt()
   
   if run_options['plot_iteration']:
      postprocess.plot_optimization_history(coupling_opt, include_func = False, norm_with_start=True)

if run_options['plot_coupled']:
   postprocess.plot_eigenmodes_3D(beam, 
                                 opt_targets = coupling_opt.targets, 
                                 initial=coupling_opt.inital, 
                                 dofs_to_plot=['y','a'],#,'z'
                                 max_normed=False, 
                                 do_rad_scale=True,
                                 number_of_modes =1,
                                 opt_params=optimization_parameters,
                                 plot_weights_in_title = False,
                                 fig_title ='after coupling opt ' + var_to_optimize,
                                 filename_for_save = 'after coupling opt ' + var_to_optimize + ' mass_not')

# ===========================
# READJUSTING FREQUENCIES
# have to set all the optimized parameters 
coupled_beam = beam.update_optimized_parameters(coupling_opt.optimized_design_params)

if run_options['dynamic_analysis']:
   coupled_dynamic_analysis = DynamicAnalysis(coupled_beam, analysis_parameters)
   coupled_dynamic_analysis.solve()
   postprocess.plot_dynamic_results(coupled_dynamic_analysis, dof_label = 'g', node_id = 3, result_variable = 'reaction')
   postprocess.plot_fft(dof_label = 'g', dynamic_analysis= coupled_dynamic_analysis)

if run_options['static_analysis']:
   coupled_beam.static_analysis_solve(apply_mean_dynamic=True, direction='all')
   postprocess.plot_static_result(coupled_beam, load_type='mean', dofs_to_plot=['y','a'], do_rad_scale=True, save_suffix='couple')
#postprocess.plot_eigenmodes_3D(coupled_beam, number_of_modes=3, dofs_to_plot=['y','a','z'], max_normed=False, fig_title='reinitialized coupled beam')


print ('\nOptimization of frequency after coupling')
freq_opt = Optimizations(model=coupled_beam)
freq_opt.adjust_sway_z_stiffness_for_target_eigenfreq(parameters['eigen_freqs'][0], 
                                                          target_mode = 0,
                                                          print_to_console=True)
# freq_opt.adjust_torsional_stiffness_for_target_eigenfreq(optimization_parameters['caarc_freqs'][2], 
#                                                           target_mode = 2,
#                                                           print_to_console=False)

if run_options['static_analysis']:
   coupled_beam.static_analysis_solve(apply_mean_dynamic=True, direction='all')
   static_deformation_couple_read = coupled_beam.static_deformation.copy()
   postprocess.plot_static_result(coupled_beam, init_deform = static_deformation_init,
                                    load_type='mean', dofs_to_plot=['y','a'],
                                    do_rad_scale=True, save_suffix='couple_re')

# ===============================
# PLOTS AFTER OPTIMIZATIONS
if run_options['plot_coupled_readjusted']:
   postprocess.plot_eigenmodes_3D(coupled_beam, 
                                 opt_targets = coupling_opt.targets, 
                                 initial=coupling_opt.inital, 
                                 dofs_to_plot=['y','a'],#,'z'
                                 max_normed=False, 
                                 do_rad_scale = True,
                                 number_of_modes = 3,
                                 opt_params=optimization_parameters,
                                 plot_weights_in_title = False,
                                 include_caarc = False,
                                 use_caarc_fitted = False,
                                 fig_title='after coupling and readjustment of freqeuncies ' + var_to_optimize,
                                 filename_for_save= 'after coupling and freq readjustment ' + var_to_optimize)

if optimization_parameters['save_optimized_parameters']:
   utilities.save_optimized_beam_parameters(coupled_beam, fname='coupled_beam')
# =============================
# OBJECTIVE FUNCTION PLOTS 
if run_options['plot_obj_func']:
   postprocess.plot_objective_function_3D(optimization_object= coupling_opt, evaluation_space_x=[-2, 2, 0.5], evaluation_space_y=[-2, 2, 0.5],
                                          include_opt_history=True, filename_for_save='objective func next ' + var_to_optimize)

#postprocess.plot_objective_function_2D(coupling_opt, evaluation_space = [-8,8,0.01],design_var_label='k_' + var_to_optimize)
#postprocess.plot_objective_function_2D(optimization)
