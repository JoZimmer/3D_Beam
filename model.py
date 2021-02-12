import numpy as np 
from scipy import linalg

from bernoulli_element import BernoulliElement

class BeamModel(object):

    def __init__(self, parameters):

        self.shear_only = parameters['shear_only']
        self.decouple = parameters['decouple']
        if self.shear_only:
            self.decouple = True

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

        #matrices
        self.dofs_of_bc = self.parameters['bc']
        self.k = np.zeros((self.n_nodes * self.n_dofs_node,
                            self.n_nodes * self.n_dofs_node))

        self.m = np.zeros((self.n_nodes * self.n_dofs_node,
                            self.n_nodes * self.n_dofs_node))
        #initialize also computational matrices -> maybe not necesary?                 
        self.comp_k = np.zeros((self.n_nodes * self.n_dofs_node,
                            self.n_nodes * self.n_dofs_node))

        self.comp_m = np.zeros((self.n_nodes * self.n_dofs_node,
                            self.n_nodes * self.n_dofs_node))

        # includes BC application
        self.calculate_and_assemble_global_matrices()
        #calculations
        #self.eigenvalue_solve()

        #Results
        self.eigenfreqs = None
        self.eigenform = np.zeros(self.n_nodes * self.n_dofs_node)
        self.static_deformation = {}

    # create n_elem Bernoulli elements and collect them in a list
    def initialize_elements(self):

        lx_i = self.parameters['lx_total_beam'] / self.n_elems

        for i in range(self.n_elems):
            # NOTE: not sure if it make sence to pass the shear 
            # and decouple to the element construction or if better doing it here
            e = BernoulliElement(self.parameters, lx_i ,i, self.shear_only, self.decouple)
            self.elements.append(e)

        self.nodal_coordinates['x0'] = np.zeros(self.n_nodes)
        self.nodal_coordinates['y0'] = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            self.nodal_coordinates['x0'][i] = i * lx_i

    def calculate_and_assemble_global_matrices(self):

        for element in self.elements:

            k_el = element.get_stiffness_matrix()
            m_el = element.get_mass_matrix()

            start = self.n_dofs_node * element.index
            end = start + self.n_dofs_node * self.nodes_per_elem

            self.k[start: end, start: end] += k_el
            self.m[start: end, start: end] += m_el
        print (self.k)

        self.comp_k = self.apply_bc_by_reduction(self.k)
        self.comp_m = self.apply_bc_by_reduction(self.m)

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

    def eigenvalue_solve(self):
        self.eig_values_raw, self.eigen_modes_raw = linalg.eigh(self.comp_k, self.comp_m)

        self.eig_values = np.sqrt(np.real(self.eig_values_raw))
        self.eig_freqs = self.eig_values / 2. / np.pi #rad/s
        self.eig_pers = 1 / self.eig_freqs

    def static_analysis_solve(self, load_vector):

        load = self.apply_bc_by_reduction(load_vector, axis='row_vector')

        #missing ground node -> get bc by extension
        static_result = np.linalg.solve(self.comp_k, load)
        static_result = self.recuperate_bc_by_extension(static_result, 'row_vector')
        for i, label in enumerate(self.dof_labels):
            self.static_deformation[label] = static_result[i::len(self.dof_labels)]

    def postprocess(self):
        pass