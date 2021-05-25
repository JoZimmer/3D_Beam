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
          'figure.dpi': 80,
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

# # 2D 

def plot_static_result(beam_model, load_type = 'single' ,dofs_to_plot = ['y'], analytic = True):
    fig, ax = plt.subplots(figsize=(5,2), num='static results')

    ax.plot(beam_model.nodal_coordinates['x0'],
            beam_model.nodal_coordinates['y0'],
            label = 'structure',
            color = 'grey',
            linestyle = '--')

    print ('\nStatic results:')
    if analytic:
        w_analytic = utilities.analytic_function_static_disp(beam_model.parameters, np.arange(0,beam_model.parameters['lx_total_beam']+1))
        ax.plot(np.arange(0,len(w_analytic)),
                    w_analytic,
                    label = 'analytic',
                    color = 'k',
                    linestyle = '--')
        print ('  w_max ist analytic: ', w_analytic[-1])

    for d_i, dof in enumerate(dofs_to_plot):
        ax.plot(beam_model.nodal_coordinates['x0'],
                beam_model.static_deformation[dof],
                label = 'final ' + dof + ' disp; tip: ' + '{0:.2e}'.format(beam_model.static_deformation[dof][-1][0]),
                color = COLORS[d_i])
            
    print ('  w_max ist beam:     ', beam_model.static_deformation['y'][-1][0])
    ax.legend()
    ax.grid()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    # if beam_model.shear_only:
    #     variant = ' - shear_only'
    # elif beam_model.decouple:
    #     variant = ' - y & g decoupled'
    # else:
    variant = ''
    ax.set_title('static displacement for ' + load_type + ' load' + variant)
    plt.show()

def plot_eigenmodes(beam_model,  fit = False, analytic = True, number_of_modes = 3):
    
    fig, ax = plt.subplots(figsize=(5,2.2), num='eigenmode results')

    ax.plot(beam_model.nodal_coordinates['x0'],
            beam_model.nodal_coordinates['y0'],
            label = 'structure',
            color = 'grey',
            linestyle = '--')

    f_j = utilities.analytic_eigenfrequencies(beam_model)
    y_analytic = utilities.analytic_eigenmode_shapes(beam_model)

    for i in range(number_of_modes):
        x = beam_model.nodal_coordinates['x0']
        y = utilities.check_and_flip_sign_array(beam_model.eigenmodes['y'][i]) 
        ax.plot(x,
                y,
                label = 'mode ' + str(i+1) + ' freq ' + str(round(beam_model.eigenfrequencies[i],3)),
                linestyle = '-',
                color = COLORS[i])

        if analytic:
            y = utilities.check_and_flip_sign_array(y_analytic[i])
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
            y_fitted = utilities.check_and_flip_sign_array(poly(x))
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

# # 3D

def plot_eigenmodes_3D(beam_model, opt_targets = None , initial = None, 
                        number_of_modes = 3, dofs_to_plot = ['y','z','a'],
                        max_normed = True, do_rad_scale =False, opt_params=None,
                        fig_title = ''):

    norm = 1
    rad_scale = np.sqrt(beam_model.parameters['cross_section_area'])
    weights = ''
    if opt_params:
        weights = '\n' + r'$weights:\,{}$'.format(str(opt_params['weights']))

    if number_of_modes == 1:
        fig, ax = plt.subplots(ncols = number_of_modes, figsize=(2,3.5), num='eigenmode results')
        fig.suptitle(fig_title)
        x = beam_model.nodal_coordinates['x0']
        ax.plot( beam_model.nodal_coordinates['y0'],
                    x,
                    label = r'$structure$',
                    color = 'grey',
                    linestyle = '--')
        ax.set_title(r'$mode\,1$ '+'\n' +r'$Frequency$: ' + r'${}$'.format(str(round(beam_model.eigenfrequencies[0],3))) + weights)

        for d_i, dof in enumerate(dofs_to_plot):
            scale=1.0
            if do_rad_scale:
                if dof in ['a','b','g']:
                    scale = rad_scale
                else:
                    scale = 1.0
            y = utilities.check_and_flip_sign_array(beam_model.eigenmodes[dof][0])

            if max_normed:
                norm = 1/max(y)
            
            if opt_targets:
                if dof in opt_targets.keys():
                    y2 = opt_targets[dof] *scale
                    ax.plot(y2*norm,
                                x,
                                label =  r'${}$'.format(dof) + r'$_{target}$',
                                linestyle = '--',
                                color = COLORS[d_i])
            if initial:
                if dof in initial.keys():
                    y3 = initial[dof]
                    ax.plot(y3*norm*scale,
                                x,
                                label =  r'${}$'.format(dof) + r'$_{inital}$',
                                linestyle = ':',
                                color = COLORS[d_i])
            ax.plot(y*norm*scale,
                        x,
                        label = r'${}$'.format(dof) + r'$_{max}:$' + '{0:.2e}'.format(max(y)),#'max: ' +
                        linestyle = '-',
                        color = COLORS[d_i])
        ax.legend()
        ax.grid()
        ax.set_xlabel(r'deflection')
        ax.set_ylabel(r'x [m]') 

        ratio = max(utilities.check_and_flip_sign_array(beam_model.eigenmodes['a'][0])) / max(utilities.check_and_flip_sign_array(beam_model.eigenmodes['y'][0]))
        ax.plot(0,0, label = r'$a_{max}/y_{max}: $' + str(round(ratio,3)))    
        ax.legend()

    else:
        fig, ax = plt.subplots(ncols = number_of_modes, figsize=(5,4), num='eigenmode results')
        fig.suptitle(fig_title)

        for i in range(number_of_modes):
            x = beam_model.nodal_coordinates['x0']
            ax[i].plot( beam_model.nodal_coordinates['y0'],
                        x,
                        label = r'$structure$',
                        color = 'grey',
                        linestyle = '--')
            ax[i].set_title(r'$mode$ ' + str(i+1) + '\n' +r'$Frequency$: ' + r'${}$'.format(str(round(beam_model.eigenfrequencies[i],3))) + weights)

            
            for d_i, dof in enumerate(dofs_to_plot):
                scale=1.0
                if do_rad_scale:
                    if dof in ['a','b','g']:
                        scale = rad_scale
                    else:
                        scale = 1.0
                y = utilities.check_and_flip_sign_array(beam_model.eigenmodes[dof][i])

                if max_normed:
                    norm = 1/max(y)
                if i == 0:
                    if opt_targets:
                        if dof in opt_targets.keys():
                            y2 = opt_targets[dof]*scale
                            ax[i].plot(y2*norm,#*scale,
                                        x,
                                        label = r'${}$'.format(dof) + r'$_{target}$',
                                        linestyle = '--',
                                        color = COLORS[d_i])
                    if initial:
                        if dof in initial.keys():
                            y3 = initial[dof]
                            ax[i].plot(y3*norm*scale,
                                        x,
                                        label = r'${}$'.format(dof) + r'$_{initial}$',
                                        linestyle = ':',
                                        color = COLORS[d_i])
                ax[i].plot(y*norm*scale,
                            x,
                            label = r'${}$'.format(dof) + r'$_{max}:$ ' + '{0:.2e}'.format(max(y)),
                            linestyle = '-',
                            color = COLORS[d_i])

            
            
            ax[i].legend()
            ax[i].grid()
            ax[i].set_xlabel(r'$deflection$')
            ax[0].set_ylabel(r'$x \, [m]$') 

        ratio = max(utilities.check_and_flip_sign_array(beam_model.eigenmodes['a'][0])) / max(utilities.check_and_flip_sign_array(beam_model.eigenmodes['y'][0]))
        ax[0].plot(0,0, label = r'$a_{max}/y_{max} = $' + str(round(ratio,3)))    
        ax[0].legend()
        #plt.tight_layout()
    plt.show()

# # OPTIMIZATIONS

def plot_objective_function_2D(optimization_object, design_var_label = 'design variable'):
    fig, ax = plt.subplots()

    objective_function = optimization_object.optimizable_function
    #x = np.arange(14000,19000)
    x = np.arange(0, 10, 0.01)
    result = np.zeros(len(x))
    for i, val in enumerate(x):
        result[i] = objective_function(val)
    
    # extreme = max(result)
    # idx = np.argwhere(result == extreme)
    ax.plot(x, result)#, label = 'max yg: '+str(x[idx]))

    if optimization_object.final_design_variable:
        ax.vlines(optimization_object.final_design_variable, 0, 1,#max(result), 
                    label = 'optimized variable: ',# + str(round(optimization_object.final_design_variable, 2)), 
                    color = 'r', 
                    linestyle ='--')
    ax.set_title('objective function')
    ax.set_xlabel('values of ' + design_var_label )
    ax.set_ylabel(r'$ f = \sum w_{i} * e_{i} ^{2}$')
    ax.grid()
    ax.legend()
    plt.show()

def plot_objective_function_2D_old(beam_model):
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

def plot_objective_function_3D(optimization_object):
    '''
    !!! be careful when this is called and what the variables in the beam object are at this time 
    target ?! correct ?
    '''
    fig = plt.figure(num='objective_func', figsize=(5,3))
    #ax = fig.add_subplot(122, projection='3d')
    ax1 = fig.add_subplot(111)

    objective_function = optimization_object.optimizable_function

    x = np.arange(-10,10,0.1)#(67000, 69000)[::10]
    y = np.arange(-10,10,0.1)#(56000, 58000)[::10]

    if x.shape != y.shape:
        raise Exception('shape of x and y input parameters must be the same for 3D plotting')
    x,y = np.meshgrid(x,y)
    z = np.zeros((len(x),len(y)))
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):

            z[i][j] = objective_function((x[i][j], y[i][j]))


    levels = np.linspace(0.1, 0.75, 2000)

    cs = ax1.contourf(x,y,z, 50, cmap = 'viridis')
    l = list(cs.levels)
    level_lines = l[:2] + l[2:10:2]
    cs2 = ax1.contour(cs, levels = level_lines, colors = 'r')

    cs_bar = fig.colorbar(cs,  shrink=0.5, aspect=20, ax = ax1)
    cs_bar.add_lines(cs2)
    # surf = ax.plot_surface(x,y,z,
    #                     #cmap= cm.coolwarm,
    #                     rstride=1, cstride=1, 
    #                     linewidth=0,
    #                     antialiased=False,
    #                     vmin = z.min(), vmax = z.max())

    #ax.plot_wireframe(x,y,z)
    #cbar = fig.colorbar(surf, shrink=0.5, aspect=20, ax = ax)#,extend={'min','max'})

    
    # ax.set_xlabel('k_ya')
    # ax.set_ylabel('k_ga')
    # ax.set_zlabel(r'$ f = \sum^{3} w_{i} * e_{i} ^{2}$')
    # ax.grid()

    #ax1.set_title('objective function with weights: ' + str(optimization_object.weights))
    ax1.set_xlabel('k_ya')
    ax1.set_ylabel('k_ga')
    cs_bar.ax.set_xlabel(r'$ f = \sum^{3} w_{i} * e_{i} ^{2}$')
    ax1.grid( linestyle='--')
    #ax.legend()
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
