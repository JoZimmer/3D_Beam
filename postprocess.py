import matplotlib.pyplot as plt 
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #could be used in stead of projection = '3d'
from matplotlib import cm

import utilities


def cm2inch(cm_value):
    return cm_value/2.54
# paper size for a4 landscape
# width = cm2inch(29.7)
# height = cm2inch(21)

# custom rectangle size for figure layout
cust_rect = [0.05, 0.05, 0.95, 0.95]

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# options
# for customizing check https://matplotlib.org/users/customizing.html
width = cm2inch(16)
height = cm2inch(11.0)
# Direct input
plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
# Options
params = {'text.usetex': True,
          'font.size': 10,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          'figure.titlesize': 10,
          'figure.figsize': (width, height),
          'figure.dpi': 300,
          'axes.titlesize': 10,
          'axes.labelsize': 10,
          'axes.grid': 'on',
          'axes.grid.which': 'both',
          'axes.xmargin': 0.05,
          'axes.ymargin': 0.05,
          'lines.linewidth': 1,
          'lines.markersize': 3,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'ytick.minor.visible': 'true',
          'xtick.minor.visible': 'true',
          'grid.linestyle': '-',
          'grid.linewidth': 0.5,
          'grid.alpha': 0.3,
          'legend.fontsize': 10,
          'savefig.dpi': 300,
          'savefig.format': 'pdf',
          'savefig.bbox': 'tight'
          }
#plt.rcParams.update(params)


def plot_static_result(beam_model, load_type = 'single' ,analytic = True):
    fig, ax = plt.subplots()

    if analytic:
        w_analytic = utilities.analytic_function_static_disp(beam_model.parameters, np.arange(0,beam_model.parameters['lx_total_beam']+1))
        ax.plot(np.arange(0,len(w_analytic)),
                    w_analytic,
                    label = 'analytic',
                    color = 'r',
                    linestyle = '--')
        print ('w_max ist analytic: ', w_analytic[-1])


    ax.plot(beam_model.nodal_coordinates['x0'],
            beam_model.nodal_coordinates['y0'],
            label = 'structure',
            color = 'grey',
            linestyle = '--')

    ax.plot(beam_model.nodal_coordinates['x0'],
            beam_model.static_deformation['y'],
            label = 'static displacement in y',
            color = 'tab:blue')
            
    print ('w_max ist beam: ', beam_model.static_deformation['y'][-1])
    ax.legend()
    ax.grid()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    if beam_model.shear_only:
        variant = ' - shear_only'
    elif beam_model.decouple:
        variant = ' - y & g decoupled'
    else:
        variant = ''
    ax.set_title('static displacement for ' + load_type + ' load' + variant)
    plt.show()

def plot_eigenmodes(beam_model,  fit = False, analytic = True, number_of_modes = 3):
    
    fig, ax = plt.subplots()

    ax.plot(beam_model.nodal_coordinates['x0'],
            beam_model.nodal_coordinates['y0'],
            label = 'structure',
            color = 'grey',
            linestyle = '--')

    f_j = utilities.analytic_eigenfrequencies(beam_model)
    y_analytic = utilities.analytic_eigenmode_shapes(beam_model)

    for i in range(number_of_modes):
        x = beam_model.nodal_coordinates['x0']
        y = utilities.check_and_change_sign(beam_model.eigenmodes['y'][i]) 
        ax.plot(x,
                y,
                label = 'mode ' + str(i+1) + ' freq ' + str(round(beam_model.eigenfrequencies[i],3)),
                linestyle = '-',
                color = COLORS[i])

        if analytic:
            y = utilities.check_and_change_sign(y_analytic[i])
            ax.plot(x, y, 
                    label = 'mode ' + str(i+1) + ' analytic'+ ' freq ' + str(round(f_j[i],3)),
                    linestyle = ':',
                    color = COLORS[i])

        if fit:
            def func (x_i, a, b):
                return a* np.sin(b*x_i)

            # params, pcov = curve_fit(func, x, y)
            # ax.plot(x, func(x, params[0], params[1]),
            #         label = 'fitted mode ' + str(i+1))

            poly = np.poly1d(np.polyfit(x,y,8))
            y_fitted = utilities.check_and_change_sign(poly(x))
            ax.plot(x, y_fitted,
                    label = 'mode ' + str(i+1) + ' poly fitted',
                    linestyle = '-.',
                    color = COLORS[i])
        

    ax.legend()
    ax.grid()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]') 
    ax.set_title('mode shapes of first 3 modes')
    plt.show()

# # OPTIMIZATIONS

def plot_objective_function_2D(beam_model):
    fig, ax = plt.subplots()

    objective_function = beam_model.objective_function
    #x = np.arange(14000,19000)
    x = np.arange(beam_model.bounds[0], beam_model.bounds[1])
    result = np.zeros(len(x))
    for i, val in enumerate(x):
        result[i] = objective_function(val)
    
    extreme = max(result)
    idx = np.argwhere(result == extreme)
    ax.plot(x, result, label = 'max yg: '+str(x[idx]))

    if beam_model.final_design_variable:
        ax.vlines(beam_model.final_design_variable, 0, max(result), 
                    label = 'optimized variable: ' + str(round(beam_model.final_design_variable, 1)), 
                    color = 'r', 
                    linestyle ='--')
    ax.set_title('objective function')
    ax.set_xlabel('values of design variable')
    ax.set_ylabel('result: (target - current)²/scaling')
    ax.grid()
    ax.legend()
    plt.show()

def plot_objective_function_3D(beam_model):
    '''
    !!! be careful when this is called and what the variables in the beam object are at this time 
    target ?! correct ?
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    objective_function = beam_model.objective_function
    x = np.arange(67000, 69000)[::10]
    y = np.arange(56000, 58000)[::10]
    x,y = np.meshgrid(x,y)
    z = np.zeros((len(x),len(y)))
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):

            z[i][j] = objective_function((x[i][j], y[i][j]))
    #z = objective_function(x,y)
    # z = np.zeros(len(x))
    # for i, val in enumerate():
    #     z[i] = objective_function([val[0], val[1]])

    surf = ax.plot_surface(x,y,z,
                    cmap= cm.coolwarm,
                    linewidth=0, 
                    antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title('objective function')
    ax.set_xlabel('k_yg')
    ax.set_ylabel('k_gg')
    ax.set_zlabel('result: (target - current)²/scaling')
    ax.grid()
    ax.legend()
    plt.show()

def plot_multiple_result_vectors(beam_model, vector_list):
    '''
    the computed result of the respective targets(egenform or static disp.) are tracked during optimization
    here some of the are plotted to see the developments
    '''
    fig, ax = plt.subplots()

    if beam_model.opt_params['optimization_target'] == 'static_tip_disp':
        w_analytic = utilities.analytic_function_static_disp(beam_model.parameters, np.arange(0,beam_model.parameters['lx_total_beam']+1))
        ax.plot(np.arange(0,len(w_analytic)),
                            w_analytic,
                            label = 'analytic',
                            color = 'r',
                            linestyle = '--',
                            marker = 'o')

    elif beam_model.opt_params['optimization_target'] == 'frequency':
        y = utilities.analytic_eigenmode_shapes(beam_model)[0]
        ax.plot(beam_model.nodal_coordinates['x0'],
                            y,
                            label = 'correct 1st mode',
                            color = 'r',
                            linestyle = '--',
                            marker = 'o')
        
    ax.plot(beam_model.nodal_coordinates['x0'],
            beam_model.nodal_coordinates['y0'],
            label = 'structure',
            color = 'black',
            linestyle = '--')

    if len(vector_list) > 50:
        vectors_to_plot = vector_list[::int(len(vector_list)/10)]
    else:
        vectors_to_plot = vector_list
    for i, vector in enumerate(vectors_to_plot):
        if i == 0:
            ax.plot(beam_model.nodal_coordinates['x0'],
                    vector,
                    color = 'green',
                    linestyle= '--',
                    marker = 'o',
                    label = 'initial guess')
        elif i < len(vectors_to_plot)-2:
            ax.plot(beam_model.nodal_coordinates['x0'],
                    vector,
                    color = 'grey',
                    linestyle= '--')
        elif i < len(vectors_to_plot)-1:
            ax.plot(beam_model.nodal_coordinates['x0'],
                    vector,
                    color = 'grey',
                    linestyle= '--',
                    label = 'intermediate steps')
        else:
            ax.plot(beam_model.nodal_coordinates['x0'],
                    vector,
                    color = 'tab:blue',
                    linestyle= '-',
                    label = 'last iteration')

    ax.legend()
    ax.grid()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('target during optimization ' + beam_model.opt_params['method'] + ' init: '+ str(round(beam_model.init_guess,1)) + 
                    ' scaling exponent: ' + str(beam_model.opt_params['scaling_exponent']))
    plt.show()

def plot_optimization_parameters(beam_model):
    '''
    during optimization:
    - the design variable during each iteration is tracked 
    - the results of the objective function are tracked
    -> plotting them here to see the optimization history
    '''
    fig, axes = plt.subplots(2,1)
    axes[0].plot(np.arange(0,len(beam_model.yg_values)), beam_model.yg_values)#, marker = 'o')
    axes[0].set_xlabel('iteration_number')
    axes[0].set_ylabel('design variable: k_yg')
    axes[0].grid()
    #plt.show()
    #np.arange(0,len(beam_model.results))
    axes[1].plot(beam_model.yg_values, beam_model.results)
    axes[1].set_xlabel('K_yg')
    axes[1].set_ylabel('(target - current)² / target**'+str(beam_model.opt_params['scaling_exponent']))
    axes[1].grid()
    #plt.show()

    plt.show()
