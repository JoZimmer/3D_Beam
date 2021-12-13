import numpy as np
import matplotlib.pyplot as plt
import copy
from os.path import join as os_join

from model import BeamModel
from optimizations import Optimizations
from postprocess import Postprocess
from utilities import utilities as utils
from plot_settings import plot_settings
from dynamic_analysis import DynamicAnalysis
from inputs import model_parameters
'''
coordinate system: x -> longitudinal axis, y -> perpendicular
'''
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]

width = utils.cm2inch(5)
height = utils.cm2inch(8)

latex = False
savefig = False

plot_params = plot_settings.get_params(width =width, height=height, usetex=latex, minor_ticks=False)

#plt.rcParams.update({'figure.figsize': (width, height)})
plt.rcParams.update(plot_params)

# # INITIALIZING A POSTPROCESSING OBJECT
postprocess = Postprocess(show_plots = True, savefig = savefig, savefig_latex = latex)

if postprocess.savefig or postprocess.savefig_latex:
   print ('\n****** FIGURE SAVINGS ACTIVE in CAARC B ******\n')

run_options = {
                  'use_optimized_params':False,
                  'plot_inital':True, # with ajdusted frequencies
                  'plot_iteration':False,
                  'plot_coupled':True,
                  'plot_coupled_readjusted':False,
                  'plot_obj_func':False, 
                  'dynamic_analysis':[False,'b','reaction'],
                  'static_analysis':False,
                  'save_optimized_parameters':False,
                 }

readjust = {'sway_z':True,'sway_y':False,'torsion':False}

available_optimized = ['coupled_beam_A.json','coupled_beam_B.json']

model = 'B'

# # MODEL GEOMETRIC, MATERIAL AND ELEMENT PARAMETERS
if model == 'A':
   parameters = model_parameters.parameters_A
elif model == 'B':
   parameters = model_parameters.parameters_B

# CREATE AN INITAL BEAM
beam = BeamModel(parameters, coupled=False, optimize_frequencies_init=True, use_translate_matrix=False)

# Parameter read from json files
using_optimized_params = False
if run_options['use_optimized_params']:
   with open(os_join(*['optimized_parameters', available_optimized]), 'r') as parameter_file:
      coupled_parameters = json.loads(parameter_file.read())
      using_optimized_params = True

      coupled_beam = BeamModel(coupled_parameters, optimize_frequencies_init=False, use_translate_matrix=False)

# # AVAILABLE OPTIMIZATION OPTIONS AND INPUT PARAMETER SETTINGS
opt_methods = ['Nelder-Mead', 'SLSQP', 'Powell', 'BFGS', 'L-BFGS-B','TNC', 'CG']

optimization_variables = ['kya','kga','both']

weights = [[0.33,0.33,.33],[0.5,0.5,0.],[.4,.4,.2],[.45,.45,.1],[.8,.1,.1],[.1,.8,.1],[.2,.4,.4],[0.1,0.1,0.8]]

optimization_parameters_B = {
                            'caarc_freqs_A': [0.231,0.429,0.536], 
                            'caarc_freqs_B_orig': [0.2,0.23,0.4],#[0.23,0.4,0.54], 
                            'eigen_freqs_tar':[0.2591, 0.3249, 1.3555], # Andi
                            'coupling_target':'realistic',
                            'consider_mode':0,
                            'var_to_optimize' :optimization_variables[2],#'ya'#'ga' both
                            'include_mass':False,
                            'method': opt_methods[1],
                            'init_guess':[0.2,13.5,0],#[0,10,0],# # k_ya, k_ga, m_cross
                            'bounds':((0.01, 100),(0.01, 100),(.0000001, 100)), #NOTE m seems to be corrected down always 
                            'weights':weights[1],
                            'save_optimized_parameters':False
                            }

optimization_parameters_A = {
                            'eigen_freqs_tar': [0.231,0.429,0.536],  #'caarc_freqs_A'
                            'consider_mode':0,
                            'coupling_target':'custom',
                            'var_to_optimize' :optimization_variables[2],#'ya'#'ga' both
                            'include_mass':False,
                            'method': opt_methods[0],
                            'init_guess':[0.,10.,0],#[-2.,5., 0.],#[0.1, 3] # k_ya, k_ga, m_cross
                            'bounds':((0.01, 100),(0.01, 100),(.0000001, 100)),
                            'weights':weights[1],
                            'save_optimized_parameters':False
                            }  
                                                      
#optimization_parameters = optimization_parameters_A
#optimization_parameters = optimization_parameters_B
if model == 'A':
   optimization_parameters = model_parameters.optimization_parameters_A
elif model == 'B':
   optimization_parameters = model_parameters.optimization_parameters_B

# # DYNAMIC ANALYSIS               
analysis_parameters = model_parameters.dynamic_analysis_parameters

if run_options['dynamic_analysis'][0]:
   dynamic_analysis = DynamicAnalysis(beam, parameters = analysis_parameters)
   dynamic_analysis.solve()
   dynamic_res_init = copy.deepcopy(dynamic_analysis.solver)
   postprocess.plot_dynamic_results(dynamic_analysis, dof_label = run_options['dynamic_analysis'][1], node_id = 3, 
                                    result_variable = run_options['dynamic_analysis'][2], save_suffix = 'initial', add_fft=True)
   dynamic_analysis.output_kinetic_energy(total=True)
   #postprocess.plot_fft(dof_label = 'g', dynamic_analysis = dynamic_analysis )

if run_options['static_analysis']:
   # an analysis with the uncoupled frequency tuned beam
   beam.static_analysis_solve(apply_mean_dynamic=True, direction='all')
   static_deformation_init = beam.static_deformation.copy()
   postprocess.plot_static_result(beam, load_type='mean', dofs_to_plot=['y','a'], do_rad_scale=True, save_suffix='initial')

if run_options['plot_inital']:
   postprocess.plot_eigenmodes_3D(beam, 
                                 number_of_modes = 3, 
                                 dofs_to_plot=['y','a','z'], 
                                 opt_targets = None,#utils.get_targets(beam, target= 'semi_realistic', opt_params=optimization_parameters),
                                 add_max_deform=False,
                                 max_normed=False,
                                 do_rad_scale=True, 
                                 plot_weights_in_title = False,
                                 model = model,
                                 caarc_A_only= model,
                                 fig_title= r'after initial frequency adjustments',
                                 filename_for_save = 'initial_freq_opt_no_bnds',
                                 show_legend=False)
   

# CREATE AN OPTIMIZATION
if not using_optimized_params:
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
      coupling_opt.eigen_vectorial_ya_opt(target_to_use=optimization_parameters['coupling_target'])
      
      if run_options['plot_iteration']:
         postprocess.plot_optimization_history(coupling_opt, include_func = False, norm_with_start=True)

      if run_options['plot_obj_func']:
         postprocess.plot_objective_function_3D(optimization_object= coupling_opt, evaluation_space_x=[-55, 55, 0.5], evaluation_space_y=[-30, 80, 0.5],
                                                include_opt_history=True, filename_for_save='objective_func_10_elems' + var_to_optimize, save_evaluation= True)

   
if run_options['plot_coupled']:
   postprocess.plot_eigenmodes_3D(beam, 
                                 opt_targets = coupling_opt.targets, 
                                 initial=coupling_opt.inital,#None,# 
                                 dofs_to_plot=['y','a'],#,'z'],#,'z'],#,'z'
                                 add_max_deform=False,
                                 max_normed=False, 
                                 do_rad_scale=True,
                                 number_of_modes =1,
                                 opt_params=optimization_parameters,
                                 model = model,
                                 plot_weights_in_title = False,
                                 fig_title ='after coupling opt ' + var_to_optimize,
                                 filename_for_save = 'step2_eigenmodes_coupled_ga_ya')

# ===========================
# READJUSTING FREQUENCIES
# have to set all the optimized parameters 
if not using_optimized_params:
   coupled_beam = beam.update_optimized_parameters(coupling_opt.optimized_design_params)


   print ('\nOptimization of frequency after coupling...')

   if readjust['sway_z']: 
      opt_params_readjust = {'init_guess':[10,10], 'bounds':(0.001,100),'weights':None,'consider_mode':None,'method':None}
      freq_opt = Optimizations(model=coupled_beam, optimization_parameters=opt_params_readjust)
      print ('   ...readjusting stiffness for sway_z')
      freq_opt.adjust_sway_z_stiffness_for_target_eigenfreq(parameters['eigen_freqs_tar'][0], 
                                                            target_mode = 0,
                                                            print_to_console=True)
   if readjust['sway_y']: 
      opt_params_readjust = {'init_guess':[10,10], 'bounds':(0.001,100),'weights':None,'consider_mode':None,'method':None}
      freq_opt_y = Optimizations(model=coupled_beam, optimization_parameters=opt_params_readjust)
      print ('   ...readjusting stiffness for sway_y')
      freq_opt_y.adjust_sway_y_stiffness_for_target_eigenfreq(parameters['eigen_freqs_tar'][1], 
                                                            target_mode = 1,
                                                            print_to_console=True)

   if readjust['torsion']:
      opt_params_readjust = {'init_guess':[10,10], 'bounds':(0.001,100),'weights':None,'consider_mode':None,'method':None}
      freq_opt_torsion = Optimizations(model=coupled_beam, optimization_parameters=opt_params_readjust)                                                     
      print ('   ...readjusting stiffness for sway_torsion')
      freq_opt.adjust_torsional_stiffness_for_target_eigenfreq(parameters['eigen_freqs_tar'][2], 
                                                               target_mode = 2,
                                                               print_to_console=True)


if run_options['static_analysis']:
   coupled_beam.static_analysis_solve(apply_mean_dynamic=True, direction='all')
   static_deformation_couple_read = coupled_beam.static_deformation.copy()
   postprocess.plot_static_result(coupled_beam, init_deform = static_deformation_init,
                                    load_type='mean', dofs_to_plot=['y','a'],
                                    do_rad_scale=True, save_suffix='couple_re')

if run_options['dynamic_analysis'][0]:
   coupled_dynamic_analysis = DynamicAnalysis(coupled_beam, analysis_parameters)
   coupled_dynamic_analysis.solve()
   postprocess.plot_dynamic_results(coupled_dynamic_analysis, dof_label = run_options['dynamic_analysis'][1], node_id = 3, 
                                    result_variable = run_options['dynamic_analysis'][2], init_res = dynamic_res_init, 
                                    save_suffix = 'couple_re', add_fft=True)
   coupled_dynamic_analysis.output_kinetic_energy(total=True)
   postprocess.plot_compare_energies({'uncoupled': dynamic_analysis.sum_energy_over_time, 'coupled':coupled_dynamic_analysis.sum_energy_over_time})
   #postprocess.plot_fft(dof_label = 'g', dynamic_analysis= coupled_dynamic_analysis)

# ===============================
# PLOTS AFTER OPTIMIZATIONS
if run_options['plot_coupled_readjusted']:
   postprocess.plot_eigenmodes_3D(coupled_beam, 
                                 opt_targets = coupling_opt.targets,#None,# 
                                 initial=coupling_opt.inital, #None,#
                                 dofs_to_plot=['y','a','z'],#,'z'
                                 max_normed=False, 
                                 add_max_deform=False,
                                 do_rad_scale = True,
                                 number_of_modes = 3,
                                 opt_params=optimization_parameters,
                                 plot_weights_in_title = False,
                                 model= model,
                                 include_caarc = False,
                                 use_caarc_fitted = False,
                                 fig_title='after coupling and readjustment of freqeuncies SWAY Z ' + var_to_optimize,
                                 filename_for_save= 'step3_eigenmodes_read')

if run_options['save_optimized_parameters']:
   utils.save_optimized_beam_parameters(coupled_beam, fname='coupled_beam_B_y_z_a_readj')


