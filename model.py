import numpy as np 
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.optimize import minimize, minimize_scalar
from functools import partial
import warnings

from bernoulli_element import BernoulliElement
import utilities
import utilities

num_zero = 1e-15

class BeamModel(object):

    def __init__(self, parameters, optimization_parameters = None):
        
        self.shear_only = parameters['shear_only']
        self.decouple = parameters['decouple']
        if self.shear_only:
            # if a shear beam is created also the coupling entries get 0
            self.decouple = True

        self.optimize = False
        if optimization_parameters:
            self.opt_params = optimization_parameters
            self.optimize = optimization_parameters['optimization_target']
            self.plot_opt_func = optimization_parameters['plot_opt_func']
            self.bounds = optimization_parameters['bounds']
            self.init_guess = optimization_parameters['init_guess_1']
            if self.opt_params['optimization_target'] == 'frequency_shear':
                self.init_guess = optimization_parameters['init_guess_2']
            self.use_minimize_scalar = optimization_parameters['use_minimize_scalar']
            self.method = optimization_parameters['method']
            self.objective_function = None
            self.warn_values = None
            self.final_design_variable = None

        # MATERIAL; GEOMETRIC AND ELEMENT
        self.parameters = parameters
        self.n_dofs_node = self.parameters['dofs_per_node']
        self.dof_labels = parameters['dof_labels']
        self.dofs_of_bc = self.parameters['bc']

        self.n_elems = parameters['n_elements']
        self.n_nodes = self.n_elems + 1
        self.nodes_per_elem = self.parameters['nodes_per_elem']
        self.nodal_coordinates = {}
        self.elements = []
        self.initialize_elements() 

        # STATIC LOAD
        self.load_vector = np.zeros(self.n_nodes*2)
        self.load_vector[-2] = self.parameters['static_load_magnitude']
        
        # includes BC application already
        self.calculate_and_assemble_global_matrices()
        self.eigenvalue_solve()
        #self.calculate_mass_entries_sensitivity(mode_id=0)

        # optimization after initial initialization and results
        self.yg_values = []
        self.gg_values = []
        self.results = []
        self.disp_vector = []
        if self.optimize == 'static_tip_disp':
            self.adjust_k_yg_for_static_disp()
        elif self.optimize == 'frequency':
            self.adjust_k_yg_for_frequency()
        elif self.optimize == 'frequency_shear':
            self.adjust_k_yg_k_gg_for_frequency()
        elif self.optimize == 'eigenform':
            self.norm_track = {0:[], 1:[],2:[]}
            self.adjust_k_yg_for_eigenform(mode_id=0, use_intermediate_corrections = True)

        #calculations
        #self.eigenvalue_solve()

# # ELEMENT INITIALIZATION AND MATRIX ASSAMBLAGE

    def initialize_elements(self):

        lx_i = self.parameters['lx_total_beam'] / self.n_elems

        for i in range(self.n_elems):
            # NOTE: not sure if it make sence to pass the shear 
            # and decouple to the element construction or if better doing it here
            e = BernoulliElement(self.parameters, lx_i ,i, self.shear_only, self.decouple, self.optimize)
            self.elements.append(e)

        self.nodal_coordinates['x0'] = np.zeros(self.n_nodes)
        self.nodal_coordinates['y0'] = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            self.nodal_coordinates['x0'][i] = i * lx_i

    def calculate_and_assemble_global_matrices(self):

        self.k = np.zeros((self.n_nodes * self.n_dofs_node,
                            self.n_nodes * self.n_dofs_node))

        self.m = np.zeros((self.n_nodes * self.n_dofs_node,
                            self.n_nodes * self.n_dofs_node))

        for element in self.elements:

            k_el = element.get_stiffness_matrix()
            m_el = element.get_mass_matrix()

            start = self.n_dofs_node * element.index
            end = start + self.n_dofs_node * self.nodes_per_elem

            self.k[start: end, start: end] += k_el
            self.m[start: end, start: end] += m_el

        self.comp_k = self.apply_bc_by_reduction(self.k)
        self.comp_m = self.apply_bc_by_reduction(self.m)


# # BOUNDARY CONDITIONS

    def apply_bc_by_reduction(self, matrix, axis = 'both'):
        n_dofs_total = np.arange(self.n_nodes * self.n_dofs_node)
        dofs_to_keep = list(set(n_dofs_total) - set(self.dofs_of_bc))
        
        if axis == 'both':
            ixgrid = np.ix_(dofs_to_keep, dofs_to_keep)
        # for a force vector
        elif axis == 'row_vector':
            ixgrid = np.ix_(dofs_to_keep, [0])
            matrix = matrix.reshape([len(matrix), 1])

        return matrix[ixgrid]

    def recuperate_bc_by_extension(self, matrix, axis = 'both'):
        n_dofs_total = np.arange(self.n_nodes * self.n_dofs_node)
        dofs_to_keep = list(set(n_dofs_total) - set(self.dofs_of_bc))

        if axis == 'both':
            rows = len(n_dofs_total)
            cols = rows
            ixgrid = np.ix_(dofs_to_keep,dofs_to_keep)
            extended_matrix = np.zeros((rows,cols))
        elif axis == 'row':
            rows = len(n_dofs_total)
            cols = matrix.shape[1]
            # make a grid of indices on interest
            ixgrid = np.ix_(dofs_to_keep, np.arange(matrix.shape[1]))
            extended_matrix = np.zeros((rows, cols))
        elif axis == 'column':
            rows = matrix.shape[0]
            cols = len(n_dofs_total)
            # make a grid of indices on interest
            ixgrid = np.ix_(np.arange(matrix.shape[0]), dofs_to_keep)
            extended_matrix = np.zeros((rows, cols))
        elif axis == 'row_vector':
            rows = len(n_dofs_total)
            cols = 1
            ixgrid = np.ix_(dofs_to_keep, [0])
            matrix = matrix.reshape([len(matrix), 1])
            extended_matrix = np.zeros((rows, cols))
        elif axis == 'column_vector':
            rows = len(n_dofs_total)
            cols = 1
            ixgrid = np.ix_(dofs_to_keep)
            extended_matrix = np.zeros((rows,))
        
        
        extended_matrix[ixgrid] = matrix

        return extended_matrix

# # SOLVES

    def eigenvalue_solve(self):
        '''
        fills: 
        self.eigenfrequencies as a list 
        self.eigenmodes as a dictionary with the dof as key and list as values 
        list[i] -> mode id 
        '''
        eig_values_raw, eigen_modes_raw = linalg.eigh(self.comp_k, self.comp_m)

        eig_values = np.sqrt(np.real(eig_values_raw))
        self.warn_values = {'K_yg':[], 'eigen_val':[]}
        # if self.optimize:
        #     if RuntimeWarning:
        #         self.warn_values['K_yg'].append(self.comp_k[0][2])
        #         self.warn_values['eigen_val'].append(eig_values[0])

        self.eigenfrequencies = eig_values / 2. / np.pi #rad/s
        self.eig_periods = 1 / self.eigenfrequencies

        gen_mass = np.matmul(np.matmul(np.transpose(eigen_modes_raw), self.comp_m), eigen_modes_raw)
        
        #print('\n generalized mass: \n', gen_mass)
        # numpy scales the eigenvectors to length 1.0 
        is_identiy = np.allclose(gen_mass, np.eye(gen_mass.shape[0]))
        if is_identiy:
            pass
            # print ('\n generalized mass is identity: ', is_identiy)
        else:
            raise Exception('generalized mass is not identiy')
       

        self.eigenmodes = {}
        for dof in self.dof_labels:
            self.eigenmodes[dof] = []
        #NOTE: her only fixed free boundary implemented 
        # could also be done with apply BC and recuperate BC to make it generic   
        for i in range(len(self.eigenfrequencies)):
            for j, dof in enumerate(self.dof_labels):
                self.eigenmodes[dof].append(np.zeros(self.n_nodes))
                self.eigenmodes[dof][i][1:] = eigen_modes_raw[j:,i][::len(self.dof_labels)]
                
    def static_analysis_solve(self):
        self.static_deformation = {}

        load = self.apply_bc_by_reduction(self.load_vector, axis='row_vector')

        #missing ground node -> get bc by extension
        static_result = np.linalg.solve(self.comp_k, load)
        static_result = self.recuperate_bc_by_extension(static_result, 'row_vector')
        for i, label in enumerate(self.dof_labels):
            self.static_deformation[label] = static_result[i::len(self.dof_labels)]

    # not used 
    def fit_eigenmodes(self):
        self.eigenmodes_fitted = {}
        x = self.nodal_coordinates['x0']
        for dof in self.eigenmodes:
            self.eigenmodes_fitted[dof] = []
            for mode in self.eigenmodes[dof]:
                polynom = np.poly1d(np.polyfit(x, mode, 8))
                self.eigenmodes_fitted[dof].append(polynom(x))

# # SENSITIVITIES
    
    # https://www.mdpi.com/2076-3417/10/7/2577/pdf -> sensitivites of eigenvectors
    # not working here and not used
    def calculate_mass_entries_sensitivity(self, mode_id):
        # need the raw eigenvalues and the generalized modes
        eig_values_raw, eigen_modes_raw = linalg.eigh(self.comp_k, self.comp_m)

        #eig_values = np.sqrt(np.real(eig_values_raw))
        # for m_yg_11
        dM_dp = np.zeros((self.comp_m.shape))

        # this leads to everything except the first ones to be zero
        m_i = np.zeros((4,4))
        m_i[0][1], m_i[1][0] = -1, -1
        m_i[2][3], m_i[3][2] = 1, 1

        for e in self.elements:
            start = self.n_dofs_node * e.index
            end = start + self.n_dofs_node * self.nodes_per_elem
            dM_dp[start: end, start: end] += m_i
        
        dM_dp = self.apply_bc_by_reduction(dM_dp)

        # assuming that eigenmodes raw are generalized 
        dphi_j_dm_yg = 0.
        c_jj = 0.5 * np.matmul(np.matmul(np.transpose(eigen_modes_raw[mode_id]), dM_dp), eigen_modes_raw[mode_id])
        for r in range(len(eig_values_raw)):
            if r != mode_id:
                c_jr = (np.matmul(np.matmul(np.transpose(eigen_modes_raw[r]), -1*eig_values_raw[mode_id]*dM_dp), eigen_modes_raw[mode_id])) \
                    / (eig_values_raw[mode_id] - eig_values_raw[r])

                dphi_j_dm_yg += c_jr * eigen_modes_raw[r] - c_jj * eigen_modes_raw[mode_id]
        
        self.dM_dp = dM_dp

# # OPTIMIZATION

# FOR STATIC DISPLACEMENT K_yg
    def adjust_k_yg_for_static_disp(self):

        # tip displacment evaluated at x = l 
        target_disp = utilities.analytic_function_static_disp(self.parameters, self.parameters['lx_total_beam'])

        self.objective_function = partial(self.yg_objective_function,
                                        target_disp)

        if self.use_minimize_scalar:
            minimization_result = minimize_scalar(self.objective_function,
                                              method= 'Brent',
                                              options= {'disp':True}
                                              )#,
                                              #bounds= self.bounds)

        else:
            minimization_result = minimize(self.objective_function,
                                            self.init_guess,
                                            method= self.method,
                                            options = {'disp':1}
                                            )

        print ('result for YG:', minimization_result.x[0])
        print ('error:', 60000 - minimization_result.x[0])
        #print ('message:', minimization_result.message)

    def yg_objective_function(self, target_disp, k_yg):
        
        for e in self.elements:
            if isinstance(k_yg, np.ndarray):
                if k_yg.size > 1:
                    raise Exception('design variable has two values')
                k_yg = k_yg[0]

            e.k_yg = k_yg

        #re-evaluate to have a new system
        self.calculate_and_assemble_global_matrices()

        self.static_analysis_solve()
        current_result = self.static_deformation['y'][-1]
        # do the analysis
        exponent = self.opt_params['scaling_exponent']
        self.disp_vector.append(self.static_deformation['y'])
        self.results.append(abs((target_disp - current_result)**2/target_disp**exponent))
        self.yg_values.append(k_yg)
        
        
        return abs((target_disp - current_result)**2 / target_disp**exponent)

# FOR FREQUENCY / EIGENFORM ONLY K_YG AS DESIGN VARIABLE
    def adjust_k_yg_for_frequency(self):
        # TODO: this could be done for the first three modes -> ,maybe a refinement of k_yg would be achieved
        # No mass tuning is done here however results are good ?!
        target_freq = utilities.analytic_eigenfrequencies(self)[0]

        self.objective_function = partial(self.yg_objective_function_freq,
                                        target_freq)
        
        if self.use_minimize_scalar:
            minimization_result = minimize_scalar(self.objective_function,
                                              method= 'Brent',
                                              options= {'disp':True}
                                              )#,
                                              #bounds= self.bounds)

        else:
            minimization_result = minimize(self.objective_function,
                                            self.init_guess,
                                            method= self.method,
                                            options = {'disp':1}
                                            )

        print ('result for YG:', minimization_result.x[0])
        print ('error:', 60000 - minimization_result.x[0])

    def yg_objective_function_freq(self, target_freq, k_yg):
        
        for e in self.elements:
            if isinstance(k_yg, np.ndarray):
                if k_yg.size > 1:
                    raise Exception('design variable has two values')
                k_yg = k_yg[0]

            e.k_yg = k_yg

        #re-evaluate to have a new system
        self.calculate_and_assemble_global_matrices()

        self.eigenvalue_solve()
        current_result = self.eigenfrequencies[0]
        # do the analysis
        scaling = self.opt_params['scaling_exponent'] #involved in result scaling

        result = (target_freq - current_result)**2 / target_freq**scaling

        # some collection of intermediate results
        self.disp_vector.append(self.eigenmodes['y'][0])
        self.results.append(result)
        self.yg_values.append(k_yg) 
        
        return abs((target_freq - current_result)**2 / target_freq**scaling)

    # eigenform 

    def adjust_k_yg_for_eigenform(self, mode_id, use_intermediate_corrections = True):
        '''
        if use intermediate corrections the mass is adjusted to mathc the target freuqnecy of the current mode
        '''
        # firstly for now the 1st mode shape
        if mode_id < 3:
            # analytic mode shape only hast 'y' displacements
            target = utilities.check_and_change_sign(utilities.analytic_eigenmode_shapes(self)[mode_id])*1.1
            target_freq = utilities.analytic_eigenfrequencies(self)[mode_id]

            self.objective_function = partial(self.yg_objective_function_eigenform,
                                            mode_id,
                                            target)

            minimization_result_k_yg = minimize(self.objective_function,
                                            self.init_guess,
                                            method= self.method,
                                            options = {'disp':0}
                                            )
            
            # check the frequency and adjust mass?!
            if use_intermediate_corrections:
                current_freq = round(self.eigenfrequencies[mode_id],3)
                target_round = round(utilities.analytic_eigenfrequencies(self)[mode_id], 3) 
                if round(target_freq, 3) != current_freq:
                    #self.adjust_y_stiffness_for_freq(targe) 
                    # print ('frequency error:', round(target_freq -self.eigenfrequencies[mode_id],3))
                    # print ('adjusting Iy for target frequency')
                    #self.adjust_Iy_for_target_freq(mode_id)
                    self.adjust_mass_density_for_target_freq(mode_id)

                # check if eigenform still fits and recursively adjust:
                current_norm = np.linalg.norm(target - utilities.check_and_change_sign(self.eigenmodes['y'][mode_id]))
                self.norm_track[mode_id].append(current_norm)
                # NOTE: rounding digits must be checked what makes sence
                if round(current_norm, 2) != 0:
                    # dont raise the mode id
                    for i in range(self.n_nodes-1):
                        nodal_error = (target[i+1] - utilities.check_and_change_sign(self.eigenmodes['y'][mode_id])[i+1])
                        if nodal_error >= 0.01:
                            print ('try to adjust node specific node:', str(i+1))
                            self.init_guess = minimization_result_k_yg.x[0]
                            self.adjust_k_yg_elem_for_eigenform(i, target[i+1], mode_id)

                #self.adjust_k_yg_for_eigenform(mode_id)

            # if frequency and eigenform could be adjusted go on to the next mode
            next_mode = mode_id + 1
            # setting the init guess to the result since this should always be better then a random guess
            self.init_guess = minimization_result_k_yg.x[0]
            self.adjust_k_yg_for_eigenform(next_mode)
            
            # print ('result for YG:', minimization_result.x[0])
            # print('\noptimizing next mode:')
            #print ('error:', 60000 - minimization_result.x[0])

        else: 
            # if i turned 3 the init guess is the last result
            self.final_design_variable = self.elements[0].k_yg # for plot afterwards
            print ('result for YG:', self.elements[0].k_yg)
            print ('final density:', self.elements[0].rho)

    def yg_objective_function_eigenform(self, mode_id, target, k_yg):
        for e in self.elements:
            if isinstance(k_yg, np.ndarray):
                if k_yg.size > 1:
                    raise Exception('design variable has two values')
                k_yg = k_yg[0]

            e.k_yg = k_yg

        #re-evaluate to have a new system
        self.calculate_and_assemble_global_matrices()

        self.eigenvalue_solve()
        current = utilities.check_and_change_sign(self.eigenmodes['y'][mode_id])

        scaling = np.linalg.norm(target)**self.opt_params['scaling_exponent']
        if round(scaling, 0) == 1.0:
            warnings.warn('Warning: scaling is 1.0') 
        result = np.linalg.norm((target - current)/scaling)

        # collect some intermediate information
        self.disp_vector.append(current)
        self.results.append(result)
        self.yg_values.append(k_yg)

        return result

    # intermediate adjustments called if use_intermediate_corrections = True

    def adjust_Iy_for_target_freq(self, mode_id):
        
        target = utilities.analytic_eigenfrequencies(self)[mode_id]
        initial_iy = self.parameters['Iy']

        self.objective_function = partial(self.iy_objective_function_for_freq,
                                            mode_id,
                                            initial_iy,
                                            target)

        minimization_result = minimize(self.objective_function,
                                            1.0,
                                            method= self.method,
                                            options = {'disp':0}
                                            )

        current_f = self.eigenfrequencies[mode_id]
        multiplier_result = minimization_result.x[0]
        print('adjusted f')

    def iy_objective_function_for_freq(self, mode_id, initial_iy, target, multiplier_fctr):
        for e in self.elements:
            Iy = initial_iy * multiplier_fctr[0]
            if isinstance(Iy, np.ndarray):
                if Iy.size > 1:
                    raise Exception('design variable has two values')
                Iy = Iy[0]
            e.Iy = Iy

        self.calculate_and_assemble_global_matrices()
        self.eigenvalue_solve()

        current = self.eigenfrequencies[mode_id]

        result = (target - current)**2/ target**2

        return result

    def adjust_mass_density_for_target_freq(self, mode_id):
        target = utilities.analytic_eigenfrequencies(self)[mode_id]
        initial_rho = self.parameters['material_density']

        objective_function_m = partial(self.rho_objective_function_for_freq,
                                            mode_id,
                                            initial_rho,
                                            target)

        minimization_result = minimize(objective_function_m,
                                            1.0,
                                            method= self.method,
                                            options = {'disp':0}
                                            )

        current_f = self.eigenfrequencies[mode_id]
        new_rho = minimization_result.x[0] * initial_rho
        print('adjusted f of mode', str(mode_id), 'with rho')

    def rho_objective_function_for_freq(self, mode_id, initial_rho, target, multiplier_fctr):
        for e in self.elements:
            rho = initial_rho * multiplier_fctr[0]
            if isinstance(rho, np.ndarray):
                if rho.size > 1:
                    raise Exception('design variable has two values')
                rho = rho[0]
            e.rho = rho

        self.calculate_and_assemble_global_matrices()
        self.eigenvalue_solve()

        current = self.eigenfrequencies[mode_id]

        result = (target - current)**2/ target**2

        return result

    def adjust_k_yg_elem_for_eigenform(self, elem_id, target_disp, mode_id):

        objective_function_elem = partial(self.yg_elem_objective_function_eigenform,
                                            elem_id,
                                            mode_id,
                                            target_disp)

        minimization_result_k_yg_elem = minimize(objective_function_elem,
                                            self.init_guess,
                                            method= self.method,
                                            options = {'disp':0}
                                            )

    def yg_elem_objective_function_eigenform(self, elem_id, mode_id, target_disp, k_yg):
        
        if isinstance(k_yg, np.ndarray):
            if k_yg.size > 1:
                raise Exception('design variable has two values')
            k_yg = k_yg[0]

        self.elements[elem_id].k_yg = k_yg
        
        #re-evaluate to have a new system
        self.calculate_and_assemble_global_matrices()

        self.eigenvalue_solve()
        current = utilities.check_and_change_sign(self.eigenmodes['y'][mode_id])

        scaling = np.linalg.norm(target_disp)**self.opt_params['scaling_exponent']
        if round(scaling, 0) == 1.0:
            warnings.warn('scaling is 1.0') 

        #result = np.linalg.norm((target - current)/scaling)
        result = (target_disp - current[elem_id+1])**2/scaling

        return result

# # FOR FREQUENCY K_YG AND K_GG DESIGN VARIABLE

    def adjust_k_yg_k_gg_for_frequency (self):
        target_freqs = utilities.analytic_eigenfrequencies(self)

        for mode_id, target_freq in enumerate(target_freqs[:self.opt_params['modes_to_consider']]):
            # self is used for better utilitiesing of it
            self.objective_function = partial(self.yg_gg_objective_function_freq,
                                                target_freq,
                                                mode_id)

            minimization_result = minimize(self.objective_function,
                                            self.init_guess,
                                            method= self.method,
                                            options = {'disp':1}
                                            )
            # after first mode opt set the result to the init
            self.init_guess = (minimization_result.x[0], minimization_result.x[1])
            print ('result for YG:', minimization_result.x[0])
            print ('error:', 60000 - minimization_result.x[0])
            print ('result for GG:', minimization_result.x[1])
            print ('error:', 80000 - minimization_result.x[1])

    def yg_gg_objective_function_freq(self, target_freq, mode_id, k_shear):

        for e in self.elements:
            # if isinstance(k_yg, np.ndarray):
            #     # if k_yg.size > 1:
            #     #     raise Exception('design variable has two values')
            #     # k_yg = k_yg[0]

            e.k_yg = k_shear[0]
            e.k_gg = k_shear[1]

        #re-evaluate to have a new system
        self.calculate_and_assemble_global_matrices()

        self.eigenvalue_solve()
        current_result = self.eigenfrequencies[mode_id]
        # do the analysis
        scaling = self.opt_params['scaling_exponent'] #involved in result scaling

        result = (target_freq - current_result)**2 / target_freq**scaling
        # some collection of intermediate results
        self.disp_vector.append(self.eigenmodes['y'][mode_id])
        self.results.append(result)
        self.yg_values.append(k_shear[0]) # design variable1
        self.gg_values.append(k_shear[1]) # design variable2
        
        return abs((target_freq - current_result)**2 / target_freq**scaling)

