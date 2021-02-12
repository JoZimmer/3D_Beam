import numpy as np 

class BernoulliElement(object):

    def __init__(self, parameters, elem_length ,elem_id, is_shear, decouple):

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
      
    
    def get_stiffness_matrix(self):

        EI = self.E * self.Iy

        k_yy_11 = 12.0 * EI / self.L**3
        k_yy_12 = -k_yy_11

        k_gg_11 = 4.0 * EI / self.L
        k_gg_12 = 2.0 * EI / self.L

        k_yg = 6.0 * EI / self.L**2

        if self.decouple:
            k_yg = 0.0
            if self.is_shear:
                k_gg_11 = 0.0
                k_gg_12 = 0.0  

        # Mueller StrcutDyn Tutorial page 122
        k = np.array([[k_yy_11, -k_yg, k_yy_12, -k_yg],
                      [-k_yg, k_gg_11, k_yg, k_gg_12],
                      [k_yy_12, k_yg, k_yy_11, k_yg],
                      [-k_yg, k_gg_12, k_yg, k_gg_11]])
        return k 

    def get_mass_matrix(self):
        # Mueller StrcutDyn Tutorial page 122
        m_yy_11 = self.rho * self.L * 13/35
        m_yy_12 = self.rho * self.L * 9/70

        m_gg_11 = self.rho * self.L**3 /105
        m_gg_12 = self.rho * self.L**3 /140

        m_yg_11 = self.rho * self.L**3 * 11/210
        m_yg_12 = self.rho * self.L**3 * 13/420

        if self.decouple:
            m_yg_11 = 0.0
            m_yg_12 = 0.0
            if self.is_shear:
                m_gg_11 = 0.0
                m_gg_12 = 0.0
            
        m = np.array([[m_yy_11, -m_yg_11, m_yy_12, m_yg_12], 
                      [-m_yg_11, m_gg_11, -m_yg_12, -m_gg_12],
                      [m_yg_12, -m_gg_12, m_yy_11, m_yg_11],
                      [m_yg_12, -m_gg_12, m_yg_12, m_gg_11 ]])
       
        return m 

