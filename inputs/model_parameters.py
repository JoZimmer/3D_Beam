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
                'It': 129062.0, #NOTE often set to Iy + Iz here diminsihed it by 200 000 such that the torsion optimization worked
                'modes_to_consider': 15,
                'static_load_magnitude': -20.0,
                'dynamic_load_file': "inputs\\dynamic_force_4_nodes.npy",
                'inital_params_yg': [1.0,1.0,1.0],#[0.001,0.0012,0.0014]
                'params_k_ya': [0,0],#[1.0, 1.0, 1.0, 1.0] # omega: stiffness coupling y - a omega1: y -g
                'params_m_ya': [0.0,0.0,0.0],#[1.0, 1.0, 1.0] # omega: stiffness coupling, psi1, psi2: mass y-a, psi3: mass g-a
                'eigen_freqs':[0.20,0.23,0.4]   
             }