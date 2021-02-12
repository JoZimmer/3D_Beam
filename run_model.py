import numpy as np
from model import BeamModel
import postprocess

parameters = {
                'n_elements': 1,
                'shear_only': False,
                'decouple': False,
                'dofs_per_node': 2, #y and g  
                'dof_labels': ['y','g'],
                'material_density': 1.0, #160.0
                'E_Modul': 2,#2.861e8,
                'nu': 1.0, #3D only
                'nodes_per_elem': 2,
                'lx_total_beam': 10.0, #total length of beam 
                'cross_section_area': 1.0,
                'Iy': 1.0,
                'Iz': 1.0, #3D only
                'It': 1.0,
                'bc': [0,1] #dofs to be fixed
}

#parameters['Iy'] = 1.0 * 1.0**3 / 12 #bh³/12

static_load = np.zeros((parameters['n_elements']+1)*2)
static_load[-1] = 0.01
beam = BeamModel(parameters)
beam.static_analysis_solve(static_load)
#beam.eigenvalue_analysis()

postprocess.plot_static_result(beam)