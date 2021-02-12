import matplotlib.pyplot as plt 

def plot_static_result(beam_model):
    fig, ax = plt.subplots()

    ax.plot(beam_model.nodal_coordinates['x0'],
            beam_model.nodal_coordinates['y0'],
            label = 'structure',
            color = 'grey',
            linewidth = 5)

    ax.plot(beam_model.nodal_coordinates['x0'],
            beam_model.static_deformation['y'],
            label = 'static displacement in y',
            marker = '*')

    ax.legend()
    ax.grid()
    ax.set_xlabel('l[m]')
    ax.set_ylabel('disp[m]')
    plt.show()

def plot_eigenmodes(beam_model, number_of_modes = 3):
    pass
