import numpy as np 

class BernoulliElement(object):

    def __init__(self, parameters, elem_length ,elem_id, is_shear, decouple, optimize = False):

        self.parameters = parameters
        self.index = elem_id
        self.is_shear = is_shear
        self.decouple = decouple

        self.E = parameters['E_Modul']
        self.A = parameters['cross_section_area']
        self.rho = parameters['material_density']
        self.L = elem_length

        self.Iy = parameters['Iy']
        self.number_of_nodes = parameters['nodes_per_elem']

        self.k_yg = 1e-10 #default non zero
        self.k_gg = 1. #default non zero
            
        self.num_zero = 1e-5
    
    def get_stiffness_matrix(self):

        EI = self.E * self.Iy

        k_yy_11 = 12.0 * EI / self.L**3
        k_yy_12 = -k_yy_11

        k_gg_11 = 4.0 * EI / self.L
        k_gg_12 = 2.0 * EI / self.L

        k_yg = 6.0 * EI / self.L**2

        if self.decouple:
            k_yg = self.k_yg 
            if self.is_shear:
                # alle einträge die mit g zu tun haben 0
                # NOTE: hier erst nur k_gg_11 um es bei einem 3D optimieurngs problem zu lassen 
                k_gg_11 = self.k_gg
                #k_gg_12 = self.num_zero
                
        # Mueller StrcutDyn Tutorial page 122
        k = np.array([[k_yy_11, -k_yg, k_yy_12, -k_yg],
                      [-k_yg, k_gg_11, k_yg, k_gg_12],
                      [k_yy_12, k_yg, k_yy_11, k_yg],
                      [-k_yg, k_gg_12, k_yg, k_gg_11]])
        return k 

    def get_mass_matrix(self):
        # Mueller StrcutDyn Tutorial page 122
        m_yy_11 = self.rho * self.L * 13./35.
        m_yy_12 = self.rho * self.L * 9./70.

        m_gg_11 = self.rho * self.L**3 /105.
        m_gg_12 = self.rho * self.L**3 /140.

        m_yg_11 = self.rho * (self.L**2) * 11./210.
        m_yg_12 = self.rho * (self.L**2) * 13./420.

        # if self.decouple:
        #     m_yg_11 = self.num_zero
        #     m_yg_12 = self.num_zero
        #     # m_yy_12 = self.num_zero
        #     # m_gg_12 = self.num_zero
        #     if self.is_shear:
        #         m_gg_11 = self.num_zero
        #         #m_gg_12 = self.num_zero
                
            
        m = np.array([[m_yy_11, -m_yg_11, m_yy_12, m_yg_12], 
                      [-m_yg_11, m_gg_11, -m_yg_12, -m_gg_12],
                      [m_yy_12, -m_yg_12, m_yy_11, m_yg_11],
                      [m_yg_12, -m_gg_12, m_yg_11, m_gg_11 ]])
       
        return m 

