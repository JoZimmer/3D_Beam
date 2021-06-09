''' 
a collection of parameters to be importet in the run module
'''
# # AVAILABLE OPTIMIZATION OPTIONS AND INPUT PARAMETER SETTINGS
opt_methods = ['Nelder-Mead', 'SLSQP', 'Powell', 'BFGS', 'L-BFGS-B','TNC', 'CG']

optimization_variables = ['kya','kga','both']

weights = [[0.33,0.33,.33],[0.5,0.5,0.],[.4,.4,.2],[.45,.45,.1],[.8,.1,.1],[.1,.8,.1],[.2,.4,.4],[0.1,0.1,0.8]]

parameters_A = {  
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
                'eigen_freqs':[0.231,0.429,0.536]
             }

optimization_parameters_A = {
                            'caarc_freqs_A': [0.231,0.429,0.536], 
                            'caarc_freqs_B': [0.2,0.23,0.4], 
                            'consider_mode':0,
                            'var_to_optimize' :optimization_variables[2],#'ya'#'ga' both
                            'include_mass':False,
                            'method': opt_methods[1],
                            'init_guess':[0.,10.,0],#[-2.,5., 0.],#[0.1, 3] # k_ya, k_ga, m_cross
                            'bounds':((0.01, 100),(0.01, 100),(.0000001, 100)), #NOTE m seems to be corrected down always 
                            'weights':weights[0],
                            'save_optimized_parameters':False
                            }

parameters_B = {  
                'dimension': '3D',
                'n_elements': 3,
                'lx_total_beam': 180,
                'material_density': 160, 
                'E_Modul': 286100000.0,
                'nu': 0.1, 
                'damping_coeff': 0.025,
                'nodes_per_elem': 2,
                'cross_section_area': 30*45,
                'Iy': 227812.5,
                'Iz': 101250.0,
                'I_param':120000.0, # used for the coupling eintries to make it independet of the others -> initialy in the scale of Iz or Iy
                'It': 229062.0, #NOTE often set to Iy + Iz here diminsihed it by 100 000 such that the torsion optimization worked. 200 000 makes the coupling not worling 
                'modes_to_consider': 15,
                'static_load_magnitude': -20.0,
                'dynamic_load_file': "inputs\\dynamic_force_4_nodes.npy",
                'inital_params_yg': [1.0,1.0,1.0],#[0.001,0.0012,0.0014]
                'params_k_ya': [0,0],#[1.0, 1.0, 1.0, 1.0] # omega: stiffness coupling y - a omega1: y -g
                'params_m_ya': [0.0,0.0,0.0],#[1.0, 1.0, 1.0] # omega: stiffness coupling, psi1, psi2: mass y-a, psi3: mass g-a
                'eigen_freqs':[0.20,0.23,0.5] #[0.23,0.4,0.54]#  
             }

optimization_parameters_B = {
                            'caarc_freqs_A': [0.231,0.429,0.536], 
                            'caarc_freqs_B': [0.2,0.23,0.5], 
                            'consider_mode':0,
                            'var_to_optimize' :optimization_variables[2],#'ya'#'ga' both
                            'include_mass':False,
                            'method': opt_methods[1],
                            'init_guess':[0.,10.,0],#[-2.,5., 0.],#[0.1, 3] # k_ya, k_ga, m_cross
                            'bounds':((0.01, 100),(0.01, 100),(.0000001, 100)), #NOTE m seems to be corrected down always 
                            'weights':weights[1],
                            'save_optimized_parameters':False
                            }