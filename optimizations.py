import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize as spo
from scipy.optimize import minimize, minimize_scalar
from scipy import linalg
from functools import partial
import postprocess

#import postprocess.plot_objective_function_2D 

import utilities as utils 

class Optimizations(object):

    def __init__(self, model, optimization_parameters=None):
        
        ''' 
        an inital model is given to this object
        '''
        self.model = model
        self.opt_geometric_props = {}
        if optimization_parameters:
        
            self.opt_params = optimization_parameters
            self.consider_mode = optimization_parameters['consider_mode']
            self.method = optimization_parameters['method']
            self.weights = optimization_parameters['weights']

        self.final_design_variable = None

# BENDING OPTIMIZATIONS

    def eigen_scalar_opt_yg(self):
        #k_full_target, m_full_target = build_system_matrix([1.0,1.0,1.0])
        f_init = self.model.eigenfrequencies[self.consider_mode].copy()
        mode_init = self.model.eigenmodes.copy()

        eigenfrequencies_target =  utils.analytic_eigenfrequencies(self.model)[self.consider_mode]
        #eigenmodes_target = utils.check_and_flip_sign_array(utils.analytic_eigenmode_shapes(self.model)[self.consider_mode])

        if self.model.n_elems == 3:
            eigenmodes_target = utils.check_and_flip_sign_dict(utils.eigenmodes_target_2D_3_elems)
            if self.model.parameters['lx_total_beam'] != 45:
                raise Exception ('total Beam lenght must be 45 to match the copied target')
        elif self.model.n_elems == 10:
            eigenmodes_target = utils.check_and_flip_sign_dict(utils.eigenmodes_target_2D_10_elems)
            if self.model.parameters['lx_total_beam'] != 150:
                raise Exception ('total Beam lenght must be 150 to match the copied target')
        else:
            raise Exception('for eigen_scalar_opt only  targets for model with 3 or 10 elements are available')
        
        #mode_id = 0
        self.optimizable_function = partial(self.obj_func_eig_scalar, self.consider_mode, eigenmodes_target)

        res_scalar = minimize_scalar(self.optimizable_function, options={'gtol': 1e-6, 'disp': True})
        print('optimization result:', res_scalar.x)

    def obj_func_eig_scalar(self, mode_id, eigenmodes_target, design_param):

        self.model.build_system_matricies([design_param, 1.0, 1.0])
        self.model.eigenvalue_solve()

        eigenmodes_cur = utils.check_and_flip_sign_dict(self.model.eigenmodes)

        f1 = utils.evaluate_residual(eigenmodes_cur['y'][mode_id], eigenmodes_target['y'][mode_id])
        f2 = utils.evaluate_residual(eigenmodes_cur['g'][mode_id], eigenmodes_target['g'][mode_id])
        f = 0.67*f1**2 + 0.33*f2**2

        print('F: ', str(f))

        return f

    # # EIGENMODE AND FREQUENCY WITH 3 DESIGN VARIABLES YG

    def eigen_vectorial_opt(self):
        
        eigenfrequencies_target =  utils.analytic_eigenfrequencies(self.model)

        if self.model.n_elems == 3:
            eigenmodes_target = utils.check_and_flip_sign_dict(utils.eigenmodes_target_2D_3_elems)
            if self.model.parameters['lx_total_beam'] != 45:
                raise Exception ('total Beam lenght must be 45 to match the copied target')
        elif self.model.n_elems == 10:
            eigenmodes_target = utils.check_and_flip_sign_dict(utils.eigenmodes_target_2D_10_elems)
            if self.model.parameters['lx_total_beam'] != 150:
                raise Exception ('total Beam lenght must be 150 to match the copied target')
        else:
            raise Exception('for eigen_scalar_opt only  targets for model with 3 or 10 elements are available')

        self.optimizable_function = partial(self.obj_func_eigen_vectorial, self.consider_mode, eigenmodes_target, eigenfrequencies_target)

        # defining bounds
        bnds = ((0.001, 100),(0.001, 100),(0.001, 100))

        # alternatively inequality constraints
        cnstrts = [{'type': 'ineq', 'fun': lambda x: 100 - x[0]},
                {'type': 'ineq', 'fun': lambda x: 100 - x[1]},
                {'type': 'ineq', 'fun': lambda x: 100 - x[2]},
                {'type': 'ineq', 'fun': lambda x: x[0] - 0.001},
                {'type': 'ineq', 'fun': lambda x: x[1] - 0.001},
                {'type': 'ineq', 'fun': lambda x: x[2] - 0.001}]

        # SLSQP works with bounds
        res_scalar = minimize(self.optimizable_function, [0.12, 0.15, 0.17], method='SLSQP', bounds=bnds, tol=1e-3, options={'gtol': 1e-3, 'ftol': 1e-3, 'disp': True})

        # trust-constr runs, but not ideal
        # res_scalar = minimize(self.optimizable_function, [0.12, 0.15, 0.17], method='trust-constr', bounds=bnds, tol=1e-3, options={'gtol': 1e-3, 'disp': True})

        # Nelder-Mead, BFGS does not work
        # res_scalar = minimize(self.optimizable_function, [0.12, 0.15, 0.17], method='Nelder-Mead', tol=1e-3, options={'gtol': 1e-3, 'ftol': 1e-3, 'disp': True})
        # res_scalar = minimize(self.optimizable_function, [0.12, 0.15, 0.17], method='BFGS', tol=1e-3, options={'gtol': 1e-3, 'ftol': 1e-3, 'disp': True})

        # TNC, L-BFGS-B, Powell does not work
        # res_scalar = minimize(self.optimizable_function, [0.12, 0.15, 0.17], method='L-BFGS-B', bounds=bnds, tol=1e-2, options={'gtol': 1e-3, 'disp': True})

        # COBYLA does not work
        # res_scalar = minimize(self.optimizable_function, [0.12, 0.15, 0.17], method='COBYLA', constraints=cnstrts, tol=1e-2, options={'gtol': 1e-3, 'disp': True})

        # SLSQP works with constraints as well
        # res_scalar = minimize(self.optimizable_function, [0.12, 0.15, 0.17], method='SLSQP', constraints=cnstrts, tol=1e-3, options={'gtol': 1e-3, 'ftol': 1e-3, 'disp': True})

        # trust-constr does not work
        # res_scalar = minimize(self.optimizable_function, [0.12, 0.15, 0.17], method='trust-constr', constraints=cnstrts, tol=1e-3, options={'gtol': 1e-3, 'disp': True})

    def obj_func_eigen_vectorial(self, mode_id, eigenmodes_target, eigenfrequencies_target, design_params):

        self.model.build_system_matricies(design_params)
        self.model.eigenvalue_solve()

        eigenmodes_cur = self.model.eigenmodes
        eigenfrequencies_cur = self.model.eigenfrequencies

        f1 = utils.evaluate_residual(eigenmodes_cur['y'][mode_id], eigenmodes_target['y'][mode_id])
        f2 = utils.evaluate_residual(eigenmodes_cur['g'][mode_id], eigenmodes_target['g'][mode_id])
        f3 = utils.evaluate_residual([eigenfrequencies_cur[mode_id]], [eigenfrequencies_target[mode_id]])

        # deformation and frequency relatively more important, than rotation
        #weights = [0.4, 0.2, 0.4]

        # deformation, rotation, frequency relatiive similar importance
        weights = [0.33, 0.33, 0.33]

        gamma = 2
        components = [weights[0]*f1**gamma, weights[1]*f2**gamma, weights[2]*f3**gamma]
        f = sum(components)

        print('Design params: ', ', '.join([str(val) for val in design_params]))
        print('Components: ', ', '.join([str(val) for val in components]))
        print('Objective funtion: ', str(f))

        return f

# FOR EIGENFREQUENCIES
    def adjust_sway_y_stiffness_for_target_eigenfreq(self, target_freq, target_mode, print_to_console=False):
        '''
        displacement in z direction -> sway_y = schwingung um y - Achse
        '''
        
        initial_iy = list(e.Iy for e in self.model.elements)

        # using partial to fix some parameters for the
        self.optimizable_function = partial(self.bending_y_geometric_stiffness_objective_function,
                                            target_freq,
                                            target_mode,
                                            initial_iy)
        init_guess = 1.0

        bnds_iy = (0.001,100)  # (1/8,8)

        # minimization_result = minimize(self.optimizable_function,
        #                                init_guess,
        #                                method='L-BFGS-B',  # 'SLSQP',#
        #                                bounds=(bnds_iy, bnds_a_sz))

        min_res = minimize_scalar(self.optimizable_function, tol=1e-06)#, options={'disp':True})

        # returning only one value!
        opt_fctr = min_res.x

        # NOTE this is only for constant Iy over the height
        self.opt_geometric_props['Iy'] = [min_res.x * iy_i for iy_i in initial_iy]

        if print_to_console:
            print('INITIAL iy:', ', '.join([str(val) for val in initial_iy]))
            print()
            print('OPTIMIZED iy: ', ', '.join([str(opt_fctr * val) for val in initial_iy]))
            
            print()
            print('FACTOR: ', opt_fctr)
            print()

    def bending_y_geometric_stiffness_objective_function(self, target_freq, target_mode, initial_iy, multiplier_fctr):

        for e in self.model.elements:
            e.Iy = multiplier_fctr * initial_iy[e.index]
            # assuming a linear dependency of shear areas
            # NOTE: do not forget to update further dependencies
            e.evaluate_relative_importance_of_shear()
            e.evaluate_torsional_inertia()

        # re-evaluate
        self.model.build_system_matricies(self.model.parameters['inital_params_yg'], 
                                          self.model.parameters['inital_params_k_ya'], 
                                          self.model.parameters['inital_params_m_ya'])

        self.model.eigenvalue_solve()

        eig_freq_cur = self.model.eigenfrequencies[target_mode]
        

        return (eig_freq_cur - target_freq)**2 / target_freq**2

    def adjust_sway_z_stiffness_for_target_eigenfreq(self, target_freq, target_mode, print_to_console=False):
        '''
        sway_z = schwingung in y richtung, um z Achse
        '''
        initial_iz = list(e.Iz for e in self.model.elements)

        # using partial to fix some parameters for the
        self.optimizable_function = partial(self.bending_z_geometric_stiffness_objective_function,
                                            target_freq,
                                            target_mode,
                                            initial_iz)
        initi_guess = 1.0

        bnds_iz = spo.Bounds(0.001, 100)#(0.001, 100)  # (1/8,8)

        # minimization_result = minimize(self.optimizable_function,
        #                                initi_guess,
        #                                method ='L-BFGS-B',
        #                                bounds = bnds_iz)

        min_res = minimize_scalar(self.optimizable_function, tol=1e-06)#, options={'disp':True})

        # returning only one value!
        #opt_iz_fctr = minimization_result.x
        opt_iz_fctr = min_res.x

        self.opt_geometric_props['Iz'] = [min_res.x * iz_i for iz_i in initial_iz]
        if print_to_console:
            print('INITIAL iz:', ', '.join([str(val) for val in initial_iz]))
            print()
            print('OPTIMIZED iz: ', ', '.join(
                [str(opt_iz_fctr * val) for val in initial_iz]))
            print()
            print('FACTOR: ', opt_iz_fctr)
            print ('Final Func:', min_res.fun)
            print()

    def bending_z_geometric_stiffness_objective_function(self, target_freq, target_mode, initial_iz, multiplier_fctr):
        
        for e in self.model.elements:
            e.Iz = multiplier_fctr * initial_iz[e.index]

            # NOTE: do not forget to update further dependencies
            e.evaluate_relative_importance_of_shear()
            e.evaluate_torsional_inertia()

        # re-evaluate
        self.model.build_system_matricies(self.model.parameters['inital_params_yg'], 
                                          self.model.parameters['inital_params_k_ya'], 
                                          self.model.parameters['inital_params_m_ya'])

        self.model.eigenvalue_solve()

        eig_freq_cur = self.model.eigenfrequencies[target_mode]        # mode_type_results is an ordered list

        result = (eig_freq_cur - target_freq)**2 / target_freq**2

        return result

    def adjust_torsional_stiffness_for_target_eigenfreq(self, target_freq, target_mode, print_to_console=False):
        initial_it = list(e.It for e in self.model.elements)
        initial_ip = list(e.Ip for e in self.model.elements)

        # NOTE: single parameter optimization seems not to be enough

        # using partial to fix some parameters for the
        self.optimizable_function = partial(self.torsional_geometric_stiffness_objective_function,
                                            target_freq,
                                            target_mode,
                                            initial_it,
                                            initial_ip)

        # NOTE: some additional reduction factor so that ip gets changes less

        init_guess = (1.0, 1.0)

        # NOTE: this seems not to be enough
        # bnds_it = (1/OptimizableStraightBeam.OPT_FCTR, OptimizableStraightBeam.OPT_FCTR)
        # bnds_ip = (1/OptimizableStraightBeam.OPT_FCTR, OptimizableStraightBeam.OPT_FCTR)

        # NOTE: seems that the stiffness contribution takes lower bound, the inertia one the upper bound
        bnds_it = (1/100, 10)
        bnds_ip = (1/11, 20)

        # NOTE: TNC, SLSQP, L-BFGS-B seems to work with bounds correctly, COBYLA not
        min_res = minimize(self.optimizable_function,
                                       init_guess,
                                       method='L-BFGS-B',
                                       bounds=(bnds_it, bnds_ip),
                                       options={'disp':False})

        # returning only one value!
        opt_fctr = min_res.x
        self.opt_geometric_props['It'] = [min_res.x[0] * it_i for it_i in initial_it]
        self.opt_geometric_props['Ip'] = [min_res.x[1] * ip_i for ip_i in initial_ip]
        if print_to_console:
            print('FACTORS It, Ip: ', ', '.join([str(val) for val in opt_fctr]))
            print ('final frequency: ', self.model.eigenfrequencies[target_mode])
            print()

    def torsional_geometric_stiffness_objective_function(self, target_freq, target_mode, initial_it, initial_ip, multiplier_fctr):

        for e in self.model.elements:
            e.It = multiplier_fctr[0] * initial_it[e.index]
            e.Ip = multiplier_fctr[1] * initial_ip[e.index]

        # re-evaluate
        self.model.build_system_matricies(self.model.parameters['inital_params_yg'], 
                                          self.model.parameters['inital_params_k_ya'], 
                                          self.model.parameters['inital_params_m_ya'])

        self.model.eigenvalue_solve()
        weights = [0]

        eig_freq_cur = self.model.eigenfrequencies[target_mode]

        return (eig_freq_cur - target_freq)**2 *100# / target_freq**2

# TORSION OPTIMIZATIONS
    
    def eigen_ya_stiffness_opt(self):
        if self.model.parameters['inital_params_k_ya'] != [0.0,0.0]:
            raise Exception('inital parameters of ya are not 0 - check if sensible')

        eigenmodes_target_y = self.model.eigenmodes['y'][self.consider_mode]*0.9
        eigenmodes_target_a = np.linspace(0, eigenmodes_target_y[-1] * 0.1, eigenmodes_target_y.shape[0]) # 0.12 is the ratio of caarc tip a / tip y 1st mode
        eigenfreq_target = self.model.eigenfrequencies[self.consider_mode]

        self.inital = {'y':self.model.eigenmodes['y'][self.consider_mode],'a':self.model.eigenmodes['a'][self.consider_mode]}
        self.targets = {'y':eigenmodes_target_y, 'a':eigenmodes_target_a}

        self.optimizable_function = partial(self.obj_func_eigen_ya_stiffnes, self.consider_mode, eigenmodes_target_y, eigenmodes_target_a, eigenfreq_target)

        bounds = None
        method_scalar = 'brent'
        bounds = ((0.001, 1),(0.001, 1))#,(0.001, 100))
        if bounds:
            method_scalar = 'bounded'

        #res_scalar = minimize_scalar(self.optimizable_function, method=method, bounds= bounds, options={'gtol': 1e-6, 'disp': True})
        # SLSQP works with bounds
        res_scalar = minimize(self.optimizable_function, x0= 0.0, method=self.method, bounds=bounds, tol=1e-3, options={'gtol': 1e-3, 'ftol': 1e-3, 'disp': True})

        # SLSQP works with constraints as well
        #res_scalar = minimize(self.optimizable_function, x0 = init_guess, method='SLSQP', constraints=cnstrts, tol=1e-3, options={'gtol': 1e-3, 'ftol': 1e-3, 'disp': True})

        #print( 'final F: ', str(self.optimizable_function))

        self.final_design_variable = res_scalar.x

        print('\noptimization result for design variable:', res_scalar.x)

    def obj_func_eigen_ya_stiffnes(self, mode_id, eigenmodes_target_y, eigenmodes_target_a, eigenfreq_target, design_param):
        if isinstance(design_param, np.ndarray):
            if design_param.size == 2:
                if design_param[0] == design_param[1]:
                    design_param = design_param[0]
                else:
                    raise Exception('design parameter has 2 variables that differ')
            else:
                design_param = design_param[0]

        self.model.build_system_matricies(params_k_ya=[design_param, 0.0]) # mass params not working yet -> positive definit matrix
        self.model.eigenvalue_solve()

        eigenmodes_cur = self.model.eigenmodes
        eigenfreq_cur = self.model.eigenfrequencies[self.consider_mode]

        f1 = utils.evaluate_residual(eigenmodes_cur['y'][mode_id], eigenmodes_target_y)
        f2 = utils.evaluate_residual(eigenmodes_cur['a'][mode_id], eigenmodes_target_a)
        f3 = utils.evaluate_residual([eigenfreq_cur], [eigenfreq_target])

        weights = [0.33,0.33,0.33]

        f = weights[0]*f1**2 + weights[1]*f2**2 + weights[2] * f3**2

        #print('F: ', str(f))

        return f


    def eigen_vectorial_k_ya_opt(self):
        '''
        optimizing the stiffness coupling entries
            K_ya
            k_ga
        ''' 
        if self.model.parameters['inital_params_k_ya'] != [0.0,0.0]:
            raise Exception('inital parameters of ya are not 0 - check if the targets are still sensible')

        eigenmodes_target_y = self.model.eigenmodes['y'][self.consider_mode]*0.9
        eigenmodes_target_a = np.linspace(0, eigenmodes_target_y[-1] * 0.012, eigenmodes_target_y.shape[0]) # 0.12 is the ratio of caarc tip a / tip y 1st mode
        eigenfreq_target = self.model.eigenfrequencies[self.consider_mode]


        self.inital = {'y':self.model.eigenmodes['y'][self.consider_mode],'a':self.model.eigenmodes['a'][self.consider_mode]}
        self.targets = {'y':eigenmodes_target_y, 'a':eigenmodes_target_a}
        
        self.optimizable_function = partial(self.obj_func_eigen_vectorial_k_ya, self.consider_mode, eigenmodes_target_y, eigenmodes_target_a, eigenfreq_target)

        # defining bounds
        # NOTE: k_ya takes lower bounds than 0.1
        bnds = self.opt_params['bounds']
        init_guess = self.opt_params['init_guess']#,1.0]#[0.0, 0.0,0.0]#[0.12, 0.15, 0.17] 

        # alternatively inequality constraints
        cnstrts = [{'type': 'ineq', 'fun': lambda x: 100 - x[0]},
                    {'type': 'ineq', 'fun': lambda x: 100 - x[1]},
                    {'type': 'ineq', 'fun': lambda x: 100 - x[2]},
                    {'type': 'ineq', 'fun': lambda x: x[0] - 0.001},
                    {'type': 'ineq', 'fun': lambda x: x[1] - 0.001},
                    {'type': 'ineq', 'fun': lambda x: x[2] - 0.001}]

        # SLSQP works with bounds
        res_scalar = minimize(self.optimizable_function,
                              x0 = init_guess,
                              method=self.method,
                              bounds=bnds, 
                              options={'ftol': 1e-6, 'disp': True})

        self.optimized_design_params = res_scalar.x
        digits = 5
        # SLSQP works with constraints as well
        #res_scalar = minimize(self.optimizable_function, x0 = init_guess, method='SLSQP', constraints=cnstrts, tol=1e-3, options={'gtol': 1e-3, 'ftol': 1e-3, 'disp': True})
        print()
        print('optimized parameters:')
        print ('  k_ya:', round(res_scalar.x[0],digits), 'absolute:', round(self.model.comp_k[1][3]))
        print ('  k_ga:', round(res_scalar.x[1],digits), 'absolute:', round(self.model.comp_k[3][5]))
        print ('  m_ya:', round(res_scalar.x[2],digits+4), 'absolute m_ya_11:', round(self.model.comp_m[1][3]), 'absolute m_ya_12:', round(self.model.comp_m[1][9]))

    def obj_func_eigen_vectorial_k_ya(self, mode_id, eigenmodes_target_y, eigenmodes_target_a, eigenfreq_target, design_params):

        self.model.build_system_matricies(params_k_ya = design_params[:2], params_m_ya=[design_params[-1],design_params[-1],0])
        self.model.eigenvalue_solve()

        eigenmodes_cur = self.model.eigenmodes
        eigenfreq_cur = self.model.eigenfrequencies[mode_id]

        f1 = utils.evaluate_residual(eigenmodes_cur['y'][mode_id], eigenmodes_target_y)
        f2 = utils.evaluate_residual(eigenmodes_cur['a'][mode_id], eigenmodes_target_a)
        f3 = utils.evaluate_residual([eigenfreq_cur], [eigenfreq_target])

        weights = self.weights

        gamma = 2
        components = [weights[0]*f1**gamma, weights[1]*f2**gamma, weights[2]*f3**gamma]
        f = sum(components *1)

        # print('Design params: ', ', '.join([str(val) for val in design_params]))
        # print('Components: ', ', '.join([str(val) for val in components]))
        # print('Objective function: ', str(f))

        return f


 # MASS MATRIX OPTIMIZATIONS
    
    def mass_entries_opt_ya(self):

        target = np.eye(self.model.n_dofs_node * self.model.n_elems)
        
        self.optimizable_function = partial(self.obj_func_gen_mass, target)

        bounds = self.opt_params['bounds']#,(0.001, 100))
        init_guess = self.opt_params['init_guess']#,1.0]#[0.0, 0.0,0.0]#[0.12, 0.15, 0.17] 


        #res_scalar = minimize_scalar(self.optimizable_function, method=method, bounds= bounds, options={'gtol': 1e-6, 'disp': True})
        # SLSQP works with bounds
        res_scalar = minimize(self.optimizable_function, x0= init_guess, method=self.method, bounds=bounds, options={'ftol': 1e-5, 'disp': True})

        print ('optimizaion result:', res_scalar.x)

    def obj_func_gen_mass(self, target, design_params):
        '''
        1. design_params are psi1, psi2 -> only ya entries the rest 0
        ''' 
        self.model.build_system_matricies(params_m_ya=[design_params[0],design_params[1], 0.0])

        eig_values_raw, eigen_modes_raw = linalg.eigh(self.model.comp_k, self.model.comp_m)
        
        gen_mass_cur = np.matmul(np.matmul(np.transpose(eigen_modes_raw), self.model.comp_m), eigen_modes_raw)
        
        f1 = utils.evaluate_residual(gen_mass_cur, target)

        return f1**2

