import numpy as np 
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.optimize import minimize, minimize_scalar
from functools import partial

from bernoulli_element import BernoulliElement
import postprocess

num_zero = 1e-15

class BeamModel(object):

    def __init__(self, parameters, optimize = False):
        
        self.shear_only = parameters['shear_only']
        self.decouple = parameters['decouple']
        if self.shear_only:
            # if a shear beam is created also the coupling entries get 0
            self.decouple = True
        self.optimize = optimize

        # material and geometric parameters
        self.parameters = parameters
        self.n_dofs_node = self.parameters['dofs_per_node']
        self.dof_labels = parameters['dof_labels']

        self.n_elems = parameters['n_elements']
        self.n_nodes = self.n_elems + 1
        self.nodes_per_elem = self.parameters['nodes_per_elem']
        self.nodal_coordinates = {}
        self.elements = []
        self.initialize_elements() 

        # MATRICES
        self.dofs_of_bc = self.parameters['bc']
        # self.k = np.zeros((self.n_nodes * self.n_dofs_node,
        #                     self.n_nodes * self.n_dofs_node))

        # self.m = np.zeros((self.n_nodes * self.n_dofs_node,
        #                     self.n_nodes * self.n_dofs_node))

        # LOAD
        self.load_vector = np.zeros(self.n_nodes*2)
        self.load_vector[-2] = self.parameters['static_load_magnitude']
        
       
        # includes BC application already
        self.calculate_and_assemble_global_matrices()

        # optimization after initial initialization
       
        if optimize:
            self.adjust_y_g_coupling()

        #calculations
        #self.eigenvalue_solve()

# # ELEMENT INITIALIZATION AND ASSAMBLAGE

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
        eig_values_raw, eigen_modes_raw = linalg.eigh(self.comp_k, self.comp_m)

        eig_values = np.sqrt(np.real(eig_values_raw))
        self.eigenfrequencies = eig_values / 2. / np.pi #rad/s
        self.eig_periods = 1 / self.eigenfrequencies

        gen_mass = np.matmul(np.matmul(np.transpose(eigen_modes_raw), self.comp_m), eigen_modes_raw)
        
        #print('\n generalized mass: \n', gen_mass)
        # numpy scales the eigenvectors to length 1. thus the eigen_modes_raw are always already mass normalized
        is_identiy = np.allclose(gen_mass, np.eye(gen_mass.shape[0]))
        if is_identiy:
            print ('\n generalized mass is identity: ', is_identiy)
        else:
            raise Exception('generalized mass is not identiy')
       

        self.eigenmodes = {}
        for dof in self.dof_labels:
            self.eigenmodes[dof] = []
        #NOTE: her only fixed free boundary implemented    
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

# # OPTIMIZATION

    def plot_objective_function(self, objective_function):
        fig, ax = plt.subplots()

        x = np.arange(15000,20000)
        result = np.zeros(len(x))
        for i, val in enumerate(x):
            result[i] = objective_function(val)
        
        ax.plot(x, result)
        ax.set_title('objective function')
        ax.set_xlabel('values of design variable')
        ax.set_ylabel('result: (target - current)²')
        ax.grid()
        plt.show()


    def adjust_y_g_coupling(self):
        
        #initial_k = self.comp_k.copy()

        # tip displacment evaluated at x = l 
        target_disp = postprocess.analytic_function_static_disp(self.parameters, self.parameters['lx_total_beam'])

        objective_function = partial(self.yg_objective_function,
                                        target_disp)

        self.plot_objective_function(objective_function)

        minimization_result = minimize_scalar(objective_function,
                                              method= 'Bounded',
                                              bounds=(1, 15000))

        print ('result for YG:', minimization_result.x)
        print ('success:', minimization_result.success)
        print ('message:', minimization_result.message)

    def yg_objective_function(self, target_disp, k_yg):
        
        for e in self.elements:
            if isinstance(k_yg, np.ndarray):
                k_yg = k_yg[0]

            e.k_yg = k_yg

            
        #re-evaluate to have a new system
        self.calculate_and_assemble_global_matrices()

        self.static_analysis_solve()
        current_result = self.static_deformation['y'][-1]
        # do the analysis
        
        self.result.append((target_disp - current_result)**2)
        self.yg_values.append(k_yg)
        # print ('current YG:', k_yg)
        # print ('current difference:', target_disp - current_result)
        # print ('current result:', (target_disp - current_result)**2 )
        return ((target_disp - current_result)**2) /target_disp**2

    def postprocess(self):
        pass