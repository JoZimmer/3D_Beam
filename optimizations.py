import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize, minimize_scalar
from functools import partial
import postprocess

#import postprocess.plot_objective_function_2D 

import utilities as utils 

class Optimizations(object):

    def __init__(self, model, optimization_parameters):

        self.model = model
        self.opt_params = optimization_parameters
        self.consider_mode = optimization_parameters['consider_mode']
        self.method = optimization_parameters['method']
        self.weights = optimization_parameters['weights']

        self.final_design_variable = None



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
        self.objective_function = partial(self.obj_func_eig_scalar, self.consider_mode, eigenmodes_target)

        res_scalar = minimize_scalar(self.objective_function, options={'gtol': 1e-6, 'disp': True})
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

        self.objective_function = partial(self.obj_func_eigen_vectorial, self.consider_mode, eigenmodes_target, eigenfrequencies_target)

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
        res_scalar = minimize(self.objective_function, [0.12, 0.15, 0.17], method='SLSQP', bounds=bnds, tol=1e-3, options={'gtol': 1e-3, 'ftol': 1e-3, 'disp': True})

        # trust-constr runs, but not ideal
        # res_scalar = minimize(self.objective_function, [0.12, 0.15, 0.17], method='trust-constr', bounds=bnds, tol=1e-3, options={'gtol': 1e-3, 'disp': True})

        # Nelder-Mead, BFGS does not work
        # res_scalar = minimize(self.objective_function, [0.12, 0.15, 0.17], method='Nelder-Mead', tol=1e-3, options={'gtol': 1e-3, 'ftol': 1e-3, 'disp': True})
        # res_scalar = minimize(self.objective_function, [0.12, 0.15, 0.17], method='BFGS', tol=1e-3, options={'gtol': 1e-3, 'ftol': 1e-3, 'disp': True})

        # TNC, L-BFGS-B, Powell does not work
        # res_scalar = minimize(self.objective_function, [0.12, 0.15, 0.17], method='L-BFGS-B', bounds=bnds, tol=1e-2, options={'gtol': 1e-3, 'disp': True})

        # COBYLA does not work
        # res_scalar = minimize(self.objective_function, [0.12, 0.15, 0.17], method='COBYLA', constraints=cnstrts, tol=1e-2, options={'gtol': 1e-3, 'disp': True})

        # SLSQP works with constraints as well
        # res_scalar = minimize(self.objective_function, [0.12, 0.15, 0.17], method='SLSQP', constraints=cnstrts, tol=1e-3, options={'gtol': 1e-3, 'ftol': 1e-3, 'disp': True})

        # trust-constr does not work
        # res_scalar = minimize(self.objective_function, [0.12, 0.15, 0.17], method='trust-constr', constraints=cnstrts, tol=1e-3, options={'gtol': 1e-3, 'disp': True})

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

    # TORSION OPTS
    
    def eigen_ya_stiffness_opt(self):
        if self.model.parameters['inital_params_k_ya'] != [0.0,0.0]:
            raise Exception('inital parameters of ya are not 0 - check if sensible')

        eigenmodes_target_y = self.model.eigenmodes['y'][self.consider_mode]*0.9
        eigenmodes_target_a = np.linspace(0, eigenmodes_target_y[-1] * 0.1, eigenmodes_target_y.shape[0]) # 0.12 is the ratio of caarc tip a / tip y 1st mode
        eigenfreq_target = self.model.eigenfrequencies[self.consider_mode]

        self.inital = {'y':self.model.eigenmodes['y'][self.consider_mode],'a':self.model.eigenmodes['a'][self.consider_mode]}
        self.targets = {'y':eigenmodes_target_y, 'a':eigenmodes_target_a}

        self.objective_function = partial(self.obj_func_eigen_ya_stiffnes, self.consider_mode, eigenmodes_target_y, eigenmodes_target_a, eigenfreq_target)

        bounds = None
        method_scalar = 'brent'
        bounds = ((0.001, 1),(0.001, 1))#,(0.001, 100))
        if bounds:
            method_scalar = 'bounded'

        #res_scalar = minimize_scalar(self.objective_function, method=method, bounds= bounds, options={'gtol': 1e-6, 'disp': True})
        # SLSQP works with bounds
        res_scalar = minimize(self.objective_function, x0= 0.0, method=self.method, bounds=bounds, tol=1e-3, options={'gtol': 1e-3, 'ftol': 1e-3, 'disp': True})

        # SLSQP works with constraints as well
        #res_scalar = minimize(self.objective_function, x0 = init_guess, method='SLSQP', constraints=cnstrts, tol=1e-3, options={'gtol': 1e-3, 'ftol': 1e-3, 'disp': True})

        #print( 'final F: ', str(self.objective_function))

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
            raise Exception('inital parameters of ya are not 0 - check if sensible')

        eigenmodes_target_y = self.model.eigenmodes['y'][self.consider_mode]*0.9
        eigenmodes_target_a = np.linspace(0, eigenmodes_target_y[-1] * 0.12, eigenmodes_target_y.shape[0]) # 0.12 is the ratio of caarc tip a / tip y 1st mode
        eigenfreq_target = self.model.eigenfrequencies[self.consider_mode]


        self.inital = {'y':self.model.eigenmodes['y'][self.consider_mode],'a':self.model.eigenmodes['a'][self.consider_mode]}
        self.targets = {'y':eigenmodes_target_y, 'a':eigenmodes_target_a}
        
        self.objective_function = partial(self.obj_func_eigen_vectorial_k_ya, self.consider_mode, eigenmodes_target_y, eigenmodes_target_a, eigenfreq_target)

        # defining bounds
        bnds = ((0.001, 100),(0.001, 100))#,(0.001, 100))
        init_guess = self.opt_params['init_guess']#,1.0]#[0.0, 0.0,0.0]#[0.12, 0.15, 0.17] 

        # alternatively inequality constraints
        cnstrts = [{'type': 'ineq', 'fun': lambda x: 100 - x[0]},
                    {'type': 'ineq', 'fun': lambda x: 100 - x[1]},
                    {'type': 'ineq', 'fun': lambda x: 100 - x[2]},
                    {'type': 'ineq', 'fun': lambda x: x[0] - 0.001},
                    {'type': 'ineq', 'fun': lambda x: x[1] - 0.001},
                    {'type': 'ineq', 'fun': lambda x: x[2] - 0.001}]

        # SLSQP works with bounds
        res_scalar = minimize(self.objective_function, x0 = init_guess, method=self.method, bounds=bnds, options={'ftol': 1e-6, 'disp': True})

        # SLSQP works with constraints as well
        #res_scalar = minimize(self.objective_function, x0 = init_guess, method='SLSQP', constraints=cnstrts, tol=1e-3, options={'gtol': 1e-3, 'ftol': 1e-3, 'disp': True})

    def obj_func_eigen_vectorial_k_ya(self, mode_id, eigenmodes_target_y, eigenmodes_target_a, eigenfreq_target, design_params):

        self.model.build_system_matricies(params_k_ya = design_params)
        self.model.eigenvalue_solve()

        eigenmodes_cur = self.model.eigenmodes
        eigenfreq_cur = self.model.eigenfrequencies[mode_id]

        f1 = utils.evaluate_residual(eigenmodes_cur['y'][mode_id], eigenmodes_target_y)
        f2 = utils.evaluate_residual(eigenmodes_cur['a'][mode_id], eigenmodes_target_a)
        f3 = utils.evaluate_residual([eigenfreq_cur], [eigenfreq_target])

        # deformation and frequency relatively more important, than rotation
        #weights = [0.4, 0.2, 0.4]

        # deformation, rotation, frequency relatiive similar importance
        weights = self.weights

        # no frequency as target included yet
        #weights = [0.5,0.5] # [0.67, 0.33]

        gamma = 2
        components = [weights[0]*f1**gamma, weights[1]*f2**gamma, weights[2]*f3**gamma]
        #components = [weights[0]*f1**gamma, weights[1]*f2**gamma]
        f = sum(components *1)

        # print('Design params: ', ', '.join([str(val) for val in design_params]))
        # print('Components: ', ', '.join([str(val) for val in components]))
        # print('Objective function: ', str(f))

        return f
