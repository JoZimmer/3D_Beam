import numpy as np 
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.optimize import minimize, minimize_scalar
from functools import partial
import warnings

from bernoulli_element import BernoulliElement
import utilities
import global_definitions as GD

num_zero = 1e-15

class BeamModel(object):

    def __init__(self, parameters, optimization_parameters = None, optimize_frequencies_init = True, use_translate_matrix=False):

 
        # MATERIAL; GEOMETRIC AND ELEMENT INFORMATION
        self.parameters = parameters
        self.dim = self.parameters['dimension']
        self.n_dofs_node = GD.n_dofs_node[self.dim]
        self.dof_labels = GD.dof_lables[self.dim]
        self.dofs_of_bc = GD.dofs_of_bc[self.dim]
        self.load_id = GD.tip_load[self.dim]
        
        self.n_elems = parameters['n_elements']
        self.n_nodes = self.n_elems + 1
        self.nodes_per_elem = self.parameters['nodes_per_elem']

        self.translate_matrix = use_translate_matrix

        self.initialize_elements() 

        self.build_system_matricies(parameters['inital_params_yg'], 
                                    parameters['params_k_ya'], 
                                    parameters['params_m_ya'])
        
        self.working_params = [] # can be used to track some mathematical errors 
        self.eigenvalue_solve()

        self.optimize_frequencies_init = optimize_frequencies_init
        
        if self.optimize_frequencies_init:
            from optimizations import Optimizations

            self.init_opt = Optimizations(self)

            self.init_opt.adjust_sway_z_stiffness_for_target_eigenfreq(utilities.caarc_freqs[0], 
                                                                        target_mode = 0,
                                                                        print_to_console=False)

            self.init_opt.adjust_sway_y_stiffness_for_target_eigenfreq(utilities.caarc_freqs[1], 
                                                                    target_mode = 1,
                                                                    print_to_console=False)


            self.init_opt.adjust_torsional_stiffness_for_target_eigenfreq(utilities.caarc_freqs[2],
                                                                        target_mode = 2,
                                                                        print_to_console=False)

            

# # ELEMENT INITIALIZATION AND GLOBAL MATRIX ASSAMBLAGE

    def initialize_elements(self):

        self.nodal_coordinates = {}
        self.elements = []

        lx_i = self.parameters['lx_total_beam'] / self.n_elems

        for i in range(self.n_elems):
            e = BernoulliElement(self.parameters, elem_length = lx_i , elem_id = i)
            self.elements.append(e)

        self.nodal_coordinates['x0'] = np.zeros(self.n_nodes)
        self.nodal_coordinates['y0'] = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            self.nodal_coordinates['x0'][i] = i * lx_i

    def build_system_matricies(self, params_yg = [1.0,1.0,1.0], params_k_ya = [0.0,0.0], params_m_ya = [0.0,0.0,0.0]):
        ''' 
        assigns K_el and M_el to the object. BC is applied already
        params_yg:
            0: alpha - stiffness coupling yg
            1: beta1 - mass coupling yg1
            2: beta2 - mass coupling yg2
        params_k_ya:
            0: omega - stiffness coupling ya
            1: omega1 - stiffness coupling gamma - a
        params_m_ya:
            1: psi1: - mass couling ya_11
            2: psi2: - mass coupling ya_12
            3: psi3: - mass coupling ga_11/12 
        '''
        self.k = np.zeros((self.n_nodes * self.n_dofs_node,
                            self.n_nodes * self.n_dofs_node))

        self.m = np.zeros((self.n_nodes * self.n_dofs_node,
                            self.n_nodes * self.n_dofs_node))

        for element in self.elements:

            k_el = element.get_stiffness_matrix_var(alpha = params_yg[0], omega = params_k_ya[0], omega1 = params_k_ya[1])
            m_el = element.get_mass_matrix_var(beta1 = params_yg[1], beta2 = params_yg[2], 
                                                psi1 = params_m_ya[0], psi2 = params_m_ya[0],
                                                psi3 = params_m_ya[2])
            if self.translate_matrix:
                T = self.get_transformation_matrix(0, 100.0)
                k_el = np.matmul(np.matmul(np.transpose(T), k_el), T)

            start = self.n_dofs_node * element.index
            end = start + self.n_dofs_node * self.nodes_per_elem

            self.k[start: end, start: end] += k_el
            self.m[start: end, start: end] += m_el

        self.opt_params_mass_ya = params_m_ya
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
        elif axis == 'row':
            # make a grid of indices on interest
            ixgrid = np.ix_(dofs_to_keep, np.arange(matrix.shape[1]))

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
        self.eigen_values_raw, self.eigen_modes_raw = linalg.eigh(self.comp_k, self.comp_m)
        # try:
        #     self.eigen_values_raw, self.eigen_modes_raw = linalg.eigh(self.comp_k, self.comp_m)
        #     self.working_params.append(self.opt_params_mass_ya)
        # except np.linalg.LinAlgError:
        #     #print ('Mass matrix is not positive definite for m_ya params:')
        #     #print (self.opt_params_mass_ya)
        #     #raise Exception('Linalg Error')
        #     return 0

        # if np.any(self.eigen_values_raw < 0):
        #     odds = self.eigen_values_raw < 0.0
        #     if len(self.eigen_values_raw[odds]) > 1:
        #         print (self.eigen_values_raw[odds])

        eig_values = np.sqrt(np.real(self.eigen_values_raw))
        

        self.eigenfrequencies = eig_values / 2. / np.pi #rad/s
        self.eig_periods = 1 / self.eigenfrequencies

        gen_mass = np.matmul(np.matmul(np.transpose(self.eigen_modes_raw), self.comp_m), self.eigen_modes_raw)

        is_identiy = np.allclose(gen_mass, np.eye(gen_mass.shape[0]), rtol=1e-05)
        if not is_identiy:
            print ('generalized mass matrix:')
            for i in range(gen_mass.shape[0]):
                for j in range(gen_mass.shape[1]):
                    if gen_mass[i][j] < 1e-3:
                        gen_mass[i][j] = 0
            print (gen_mass)
            print ('\n current m_ga param: ', self.opt_params_mass_ya, '\n')
            raise Exception('generalized mass is not identiy')
    
        if not is_identiy:
            return 0

        else:
            self.working_params.append(self.opt_params_mass_ya)

        self.eigenmodes = {}
        for dof in self.dof_labels:
            self.eigenmodes[dof] = []
        # NOTE: her only fixed free boundary implemented 
        # could also be done with apply BC and recuperate BC to make it generic   
        for i in range(len(self.eigenfrequencies)):
            for j, dof in enumerate(self.dof_labels):
                self.eigenmodes[dof].append(np.zeros(self.n_nodes))
                dof_and_mode_specific = self.eigen_modes_raw[j:,i][::len(self.dof_labels)]
                self.eigenmodes[dof][i][1:] = self.eigen_modes_raw[j:,i][::len(self.dof_labels)]

        self.eigenmodes = utilities.check_and_flip_sign_dict(self.eigenmodes)
        
    def static_analysis_solve(self):
        ''' 
        static load so far defaults as tip load in y direction
        TODO: include passing a load vector
        ''' 
        load_vector = np.zeros(self.n_nodes*self.n_dofs_node)
        load_vector[self.load_id] = self.parameters['static_load_magnitude']

        self.static_deformation = {}

        load = self.apply_bc_by_reduction(self.load_vector, axis='row_vector')

        #missing ground node -> get bc by extension
        static_result = np.linalg.solve(self.comp_k, load)
        static_result = self.recuperate_bc_by_extension(static_result, 'row_vector')
        for i, label in enumerate(self.dof_labels):
            self.static_deformation[label] = static_result[i::len(self.dof_labels)]

# # RETURN OPTIMIZED PARAMETERS

    def update_optimized_parameters(self, optimization_results = None):
        ''' 
        NOTE: as it is implemented only a homogenous corss sections are updated 
        ''' 

        if self.optimize_frequencies_init:
            for param, val in self.init_opt.opt_geometric_props.items():
                self.parameters[param] = val[0]

        if optimization_results:
            for param, val in optimization_results.items():
                self.parameters[param] = list(val)

        print('\nupdated optimized params')
        return self
# # EXTRA FUNCTIONS

    def get_transformation_matrix(self,ey,ez):
        ''' 
        transform the element matrix from the shear center to the coordinate center (geometric center)
            ey: eccentricity in y direction -> coupling z displacement with torsion
            ez: eccentricity in z direction -> coupling y displacement with torsion
        '''
         
        # eigentlich auch noch ez, ey aus stockwerk oben drüber --> node 2
        T = np.identity(GD.n_dofs_node[self.dim]*2)
        T_transform_part_node1 =  np.array([[1., 0., ez],
                                            [0., 1., -ey],
                                            [0., 0., 1.]])
        T_transform_part_node2 =  np.array([[1., 0., ez],
                                            [0., 1., -ey],
                                            [0., 0., 1.]])
        
        start = GD.dof_lables[self.dim].index('y')
        end = GD.dof_lables[self.dim].index('a')+1
        T[start:end, start:end] = T_transform_part_node1

        start = GD.n_dofs_node[self.dim] + GD.dof_lables[self.dim].index('y')
        end = GD.n_dofs_node[self.dim] + GD.dof_lables[self.dim].index('a')+1
        T[start:end, start:end] = T_transform_part_node2
                                    
        return T
