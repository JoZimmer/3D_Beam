import matplotlib.pyplot as plt 
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import cm
import copy
from os.path import join as os_join

import utilities
import global_definitions as GD
import CAARC_utilities as caarc_utils

AB = 'CAARC_A' # 'CAARC_B'

dest_mode_results = 'plots\\'+AB+'\\eigenmode_results\\'
dest_objective_func = 'plots\\'+AB+'\\objective_function\\'
dest_1D_opt = 'plots\\'+AB+'\\ya_yg\\'
dest_mass = 'plots\\'+AB+'\\mass_inclusion\\'
dest_static = 'plots\\'+AB+'\\static_analysis\\'

dest_latex = 'C:\\Users\\Johannes\\LRZ Sync+Share\\MasterThesis\\Abgabe\\Text\\images\\'

greek = {'y':'y','z':'z', 'x':'x','a':r'\alpha', 'b':r'\beta', 'g':r'\gamma'}

# custom rectangle size for figure layout
cust_rect = [0.05, 0.05, 0.95, 0.95]

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
caarc_cols = ['blue','red','lime']

def cm2inch(value):
    return value/2.54

class Postprocess(object):
    def __init__(self, show_plots = True, savefig = False, savefig_latex = False):

        self.show_plots = show_plots
        self.savefig = savefig
        self.savefig_latex = savefig_latex

    # # 2D 

    def plot_static_result(self,beam_model, init_deform = None, load_type = 'single' ,
                        dofs_to_plot = ['y'], analytic = False, do_rad_scale = True, save_suffix = ''):

        if not self.savefig or not self.savefig_latex:
            title = 'static results ' + save_suffix
        else:
            title= None

        fig, ax = plt.subplots(num=title )

        ax.plot(beam_model.nodal_coordinates['y0'],
                beam_model.nodal_coordinates['x0'],
                label = 'structure',
                color = 'grey',
                linestyle = '--')

        print ('\nStatic results:')
        if analytic:
            w_analytic = utilities.analytic_function_static_disp(beam_model.parameters, np.arange(0,beam_model.parameters['lx_total_beam']+1))
            ax.plot(w_analytic,
                    np.arange(0,len(w_analytic)),
                        label = 'analytic',
                        color = 'k',
                        linestyle = '--')
            print ('  w_max ist analytic: ', w_analytic[-1])

        rad_scale = np.sqrt(beam_model.parameters['cross_section_area'])

        for d_i, dof in enumerate(dofs_to_plot):
            scale=1.0
            if do_rad_scale:
                if dof in ['a','b','g']:
                    scale = rad_scale
                else:
                    scale = 1.0
            ax.plot(beam_model.static_deformation[dof] * scale,
                    beam_model.nodal_coordinates['x0'],
                    label = dof + ' tip disp: ' + '{0:.2e}'.format(beam_model.static_deformation[dof][-1][0]),
                    color = COLORS[d_i])
            if init_deform:
                ax.plot(init_deform[dof] * scale,
                        beam_model.nodal_coordinates['x0'],
                        label = 'init ' + dof + ' tip disp: ' + '{0:.2e}'.format(init_deform[dof][-1][0]),
                        color = COLORS[d_i],
                        linestyle = ':')
                
        print ('  w_max ist beam:     ', beam_model.static_deformation['y'][-1][0])
        ax.legend(loc= 'lower right')
        ax.grid()
        ax.set_xlabel('deformation [m]')
        ax.set_ylabel('height [m]')
        ax.set_ylim(0)
        variant = ''
        ax.set_title('static displacement for ' + load_type + ' load' )
        if self.savefig:
            plt.savefig(dest_static + 'static_result_' + save_suffix)
        if self.show_plots:
            plt.show()
        plt.close()

    def plot_eigenmodes(self,beam_model,  fit = False, analytic = True, number_of_modes = 3):
        
        fig, ax = plt.subplots( num='eigenmode results')#figsize=(5,2.2),

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
        if self.show_plots:
            plt.show()

    # # 3D

    def plot_eigenmodes_3D(self,beam_model, opt_targets = None , initial = None, opt_params=None,
                            number_of_modes = 3, dofs_to_plot = ['y','z','a'], add_max_deform = True,
                            max_normed = True, do_rad_scale =False, include_caarc = False, use_caarc_fitted = False,
                            plot_weights_in_title = False, fig_title = '', filename_for_save = '0_no_name'):

        norm, c_norm = 1, 1
        rad_scale = np.sqrt(beam_model.parameters['cross_section_area'])
        if max_normed:
            do_rad_scale = False
        weights = ''
        if plot_weights_in_title:
            weights = '\n' + r'$weights:\,{}$'.format(str(opt_params['weights']))

        if include_caarc:
            caarc_props = caarc_utils.get_CAARC_properties()
            c_x = caarc_props['storey_level']
            if use_caarc_fitted:
                c_modes = caarc_props['eigenmodes_fitted']
            else:
                c_modes = caarc_props['eigenmodes']

        if number_of_modes == 1:
            fig, ax = plt.subplots(ncols = number_of_modes,  num='eigenmode results')#figsize=(2,3.5),
            if not self.savefig and not self.savefig_latex:
                fig.suptitle(fig_title)

            x = beam_model.nodal_coordinates['x0']
            ax.plot( beam_model.nodal_coordinates['y0'],
                        x,
                        #label = r'$structure$',
                        color = 'grey',
                        linestyle = '--')
            ax.set_title(r'$mode\,1$ '+'\n' +r'$f = $ ' + r'${}$'.format(str(round(beam_model.eigenfrequencies[0],3))) + r' $Hz$'  + weights)

            for d_i, dof in enumerate(dofs_to_plot):
                scale=1.0
                if do_rad_scale:
                    if dof in ['a','b','g']:
                        scale = rad_scale
                    else:
                        scale = 1.0
                y = utilities.check_and_flip_sign_array(beam_model.eigenmodes[dof][0])
                if include_caarc:
                    c_y = utilities.check_and_flip_sign_array(c_modes[dof][0])

                if max_normed:
                    norm = 1/max(y)
                    c_norm = 1/max(c_y)

                if initial:
                    if dof in initial.keys():
                        y3 = initial[dof]
                        ax.plot(y3*norm*scale,
                                    x,
                                    label =  r'${}$'.format(greek[dof]) + r'$_{inital}$',
                                    linestyle = ':',
                                    color = COLORS[d_i])
                if opt_targets:
                    if dof in opt_targets.keys():
                        y2 = opt_targets[dof] *scale
                        ax.plot(y2*norm,
                                    x,
                                    label =  r'${}$'.format(greek[dof]) + r'$_{target}$',
                                    linestyle = '--',
                                    color = COLORS[d_i])
                
                lab = greek[dof]
                label = r'${}$'.format(greek[dof])
                if add_max_deform:
                    label += r'$_{max}:$' + r'${0:.2e}$'.format(max(y))
                ax.plot(y*norm*scale,
                            x,
                            label = label ,#'max: ' +
                            linestyle = '-',
                            color = COLORS[d_i])
                if include_caarc:
                    ax.plot(c_y*c_norm*scale,
                            c_x,
                            label = r'${}$'.format(greek[dof]) + r'$benchmark$',#'max: ' +
                            linestyle = ':',
                            color = caarc_cols[d_i])
            ax.legend()
            
            ax.grid()
            ax.set_ylim(bottom=0)
            ax.set_xlabel(r'$defl. \, [m]$')

            #ax.set_yticklabels([])
            ax.set_ylabel(r'$x \, [m]$') 

            ratio = max(utilities.check_and_flip_sign_array(beam_model.eigenmodes['a'][0])) / max(utilities.check_and_flip_sign_array(beam_model.eigenmodes['y'][0]))
            ratio_label = r'$\alpha_{max}/y_{max} = $' + str(round(ratio,3))

            ax.legend(loc= 'lower right')

            props = dict(boxstyle='round', facecolor='white', edgecolor = 'lightgrey', alpha=0.8)

            # place a text box in upper left in axes coords
            ax.text(0.12, 0.97, ratio_label, transform=ax.transAxes, verticalalignment='top', bbox=props)

        else:
            fig, ax = plt.subplots(ncols = number_of_modes, sharey=True,  num='eigenmode results')#figsize=(5,4),
            if not self.savefig and not self.savefig_latex:
                fig.suptitle(fig_title)

            for i in range(number_of_modes):
                x = beam_model.nodal_coordinates['x0']
                ax[i].plot( beam_model.nodal_coordinates['y0'],
                            x,
                            #label = r'$structure$',
                            color = 'grey',
                            linestyle = '--')
                ax[i].set_title(r'$mode$ ' + r'${}$'.format(str(i+1)) + '\n' +r'$f=$ ' + r'${}$'.format(str(round(beam_model.eigenfrequencies[i],3))) +r' $Hz$' + weights)

                
                for d_i, dof in enumerate(dofs_to_plot):
                    scale=1.0
                    if do_rad_scale:
                        if dof in ['a','b','g']:
                            scale = rad_scale
                        else:
                            scale = 1.0
                        
                    y = utilities.check_and_flip_sign_array(beam_model.eigenmodes[dof][i])
                    if include_caarc:
                        c_y = utilities.check_and_flip_sign_array(c_modes[dof][i])

                    if max_normed:
                        norm = 1/max(y)
                        c_norm = 1/max(c_y)
                    if i == 0:
                        if initial:
                            if dof in initial.keys():
                                y3 = initial[dof]
                                ax[i].plot(y3*norm*scale,
                                            x,
                                            label = r'${}$'.format(greek[dof]) + r'$_{initial}$',
                                            linestyle = ':',
                                            color = COLORS[d_i])
                        if opt_targets:
                            if dof in opt_targets.keys():
                                y2 = opt_targets[dof]*scale
                                ax[i].plot(y2*norm,#*scale,
                                            x,
                                            label = r'${}$'.format(greek[dof]) + r'$_{target}$',
                                            linestyle = '--',
                                            color = COLORS[d_i])
                        
                    label = r'${}$'.format(greek[dof])
                    if add_max_deform:
                        label += r'$_{max}:$' + r'${0:.2e}$'.format(max(y))
                    ax[i].plot(y*norm*scale,
                                x,
                                label = label,
                                linestyle = '-',
                                color = COLORS[d_i])

                    if include_caarc:
                        ax[i].plot(c_y*c_norm*scale,
                                    c_x,
                                    label = r'${}$'.format(greek[dof]) + r'$benchmark$',#'max: ' +
                                    linestyle = ':',
                                    color = caarc_cols[d_i])
                    
                ax[i].legend(loc = 'lower right')
                ax[i].grid()
                ax[i].set_ylim(bottom=0)
                ax[i].set_xlabel(r'$defl. \, [m] $')
                ax[0].set_ylabel(r'$x \, [m]$') 

            ratio = max(utilities.check_and_flip_sign_array(beam_model.eigenmodes['a'][0])) / max(utilities.check_and_flip_sign_array(beam_model.eigenmodes['y'][0]))
            ratio_label = r'$\alpha_{max}/y_{max} = $' + str(round(ratio,3))

            # ratio_legend = ax[0].legend(loc='upper right', title = ratio_label)
            # ax[0].add_artist(ratio_legend)

            #ax[0].plot(0,0, label = r'$\alpha_{max}/y_{max} = $' + str(round(ratio,3)), linestyle = 'None')    
            ax[0].legend(loc= 'lower right')

            props = dict(boxstyle='round', facecolor='white', edgecolor = 'lightgrey', alpha=0.8)

            # place a text box in upper left in axes coords
            ax[0].text(0.2, 0.97, ratio_label, transform=ax[0].transAxes, verticalalignment='top', bbox=props)
                    
        
        if self.savefig:
            plt.savefig(dest_mode_results + filename_for_save)
            print ('\nsaved: ', dest_mode_results + filename_for_save)
        if self.savefig_latex:
            plt.savefig(dest_latex + filename_for_save)
            print('\nsaved in LATEX folder:', dest_latex + filename_for_save)
        if self.show_plots:
            plt.show()

        plt.close()

    # # OPTIMIZATIONS

    def plot_objective_function_2D(self,optimization_object, evaluation_space = [0,10, 0.01],design_var_label = 'design variable'):

        print ('\n EVALUATE AND PLOT OBJECTIVE FUNCTION\n')

        fig, ax = plt.subplots(figsize=(5,3), num='objective_func_1D')

        objective_function = optimization_object.optimizable_function
        x = np.arange(evaluation_space[0], evaluation_space[1], evaluation_space[2])
        result = np.zeros(len(x))
        for i, val in enumerate(x):
            result[i] = objective_function(val)
        
        ax.plot(x, result)
        opt_res = optimization_object.optimized_design_params
        ax.plot(opt_res, objective_function(opt_res),linestyle = 'None', marker='o',mfc='r',mec='k', ms=4, label='optimized variable ' + str(round(opt_res,4)))

        if optimization_object.optimized_design_params:
            ax.vlines(optimization_object.optimized_design_params, min(result), max(result), 
                        #label = 'optimized variable: ',# + str(round(optimization_object.final_design_variable, 2)), 
                        color = 'r', 
                        linestyle ='--')
        ax.set_title('optimizable function')
        ax.set_xlabel('values of ' + design_var_label )
        ax.set_ylabel(r'$ f = \sum w_{i} * e_{i} ^{2}$')
        ax.grid()
        ax.legend()
        # if self.savefig:
        #     plt.savefig(dest_mode_results)

        # if self.show_plots:
        #     plt.show()
        
        plt.close()

    def plot_objective_function_3D(self,optimization_object, evaluation_space_x = [-4,4,0.1], evaluation_space_y = None,
                                    include_opt_history = False, fig_label = '', filename_for_save ='0_no_name', add_3D_surf_plot = False, save_evaluation = True):
        '''
        deepcopy of the optimization is created to avoid undesired changes in the base objects of this class.
            evaluation_space[0,1]: space where the optimizable funciton shall be evaluated
            evaluation_space[2]: steps of evaluation in this space
            if space_y is not given the same space for x and y is used
        '''
        print ('\nEVALUATE AND PLOT OBJECTIVE FUNCTION...\n')
        print (filename_for_save + str(optimization_object.weights))
        optimization_obj_eval = copy.deepcopy(optimization_object)
        fig_title = 'objective_func_' + fig_label
        fig = plt.figure(num=fig_title)#, figsize=(5,3))
        if add_3D_surf_plot:
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122, projection='3d')
        else:
            ax1 = fig.add_subplot(111)
        if not self.savefig or not self.savefig_latex:
            ax1.set_title(r'objective function with weights: ' + r'${}$'.format(optimization_obj_eval.weights))

        objective_function = optimization_obj_eval.optimizable_function

        # get the optimization information before the function evaluation
        # needs explictly to be copied since the object changes during function evaluation
        if include_opt_history:
            k_ya_hist = np.copy(optimization_obj_eval.optimization_history['k_ya'])
            k_ga_hist = np.copy(optimization_obj_eval.optimization_history['k_ga'])
            func_hist = np.copy(optimization_obj_eval.optimization_history['func'])

        if not evaluation_space_y:
            evaluation_space_y = evaluation_space_x

        x = np.arange(evaluation_space_x[0],evaluation_space_x[1],evaluation_space_x[2])
        y = np.arange(evaluation_space_y[0],evaluation_space_y[1],evaluation_space_y[2])

        if x.shape != y.shape:
            raise Exception('shape of x and y input parameters must be the same for 3D plotting')
        x,y = np.meshgrid(x,y)
        z = np.zeros((len(x),len(y)))
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):

                z[i][j] = objective_function((x[i][j], y[i][j]))

        if save_evaluation:
            fname = os_join(*['objective_functions',utilities.join_whitespaced_string(filename_for_save) + str(optimization_obj_eval.weights[0])[-1:] + '.npy'])

        levels = np.linspace(0.1, 0.75, 2000)

        cs = ax1.contourf(x,y,z, 50, cmap = 'viridis')
        l = list(cs.levels)
        level_lines = l[:2] + l[2:10:2]
        cs2 = ax1.contour(cs, levels = level_lines, colors = 'r', linewidths= 0.5)

        if include_opt_history:
            ax1.plot(k_ya_hist, 
                    k_ga_hist, 
                    linestyle = '-', color = 'grey', #linewidth= # line stuff
                    marker = 'o', mfc = 'grey', mec='k', ms = 3,  # marker stuff
                    label='iterations')
            ax1.scatter(k_ya_hist[0],k_ga_hist[0], marker='o',c='lime',edgecolors='k', s = 20,label='start',zorder=5)
            ax1.scatter(k_ya_hist[-1],k_ga_hist[-1], marker='o',c='red',edgecolors='k', s= 20, label='end',zorder=5)

        
        
        if add_3D_surf_plot:
            surf = ax2.plot_surface(x,y,z,
                                cmap= 'hsv',
                                rstride=1, cstride=1, 
                                linewidth=0,
                                antialiased=False,
                                vmin = z.min(), vmax = z.max())

            ax2.plot_wireframe(x,y,z)
            cbar = fig.colorbar(surf, shrink=0.5, aspect=20, ax = ax2)#,extend={'min','max'})

            
            ax2.set_xlabel('k_ya')
            ax2.set_ylabel('k_ga')
            ax2.set_zlabel(r'$ f = \sum^{3} w_{i} * e_{i} ^{2}$')
            ax2.grid()

        ax1.set_xlabel(r'$k_{ya}$')
        ax1.set_ylabel(r'$k_{ga}$')

        ax1.set_xlim(-25,25)
        #ax1.margins(x = -0.3)
        ax1.set_ylim(-10,65)
        self.set_ax_size(utilities.cm2inch(6), utilities.cm2inch(4), ax=ax1)

        #cs_bar.ax.set_xlabel(r'$ f = (\mathbf{x})$')
        cs_bar = fig.colorbar(cs,  shrink=0.5, aspect=20, ax = ax1, pad=0.001)
        cs_bar.add_lines(cs2)
        cs_bar.set_label(r'$ f = (\mathbf{x})$')
        ax1.grid( linestyle='--')
        ax1.legend()

        if self.savefig:
            plt.savefig(dest_objective_func + filename_for_save + str(optimization_obj_eval.weights[0])[-1:])
            print ('\nsaved:',dest_objective_func + filename_for_save + str(optimization_obj_eval.weights[0])[-1:])
        if self.savefig_latex:
            plt.savefig(dest_latex + filename_for_save + str(optimization_obj_eval.weights[0])[-1:])
            print ('\nsaved:',dest_latex + filename_for_save + str(optimization_obj_eval.weights[0])[-1:])

        if self.show_plots:
            plt.show()
        plt.close()
        
        del optimization_obj_eval


    def plot_optimization_history(self, optimization_object, include_func, norm_with_start = True):

        fig = plt.figure(num='opt_history')#, figsize=(cm2inch(7.3), cm2inch(4.8))

        for key, val in optimization_object.optimization_history.items():
            if not include_func and key == 'func':
                continue
            if key == 'iter':
                continue
            if norm_with_start:
            #     if val[0] == 0:
            #         val[0] = 0.01
                val_norm = [val_i - val[0] for val_i in val]
            else:
                val_norm = val
            plt.plot(np.arange(1,len(val)+1), val_norm, label=utilities.prepare_string_for_latex(key))
            plt.xlabel('Iteration')
            plt.xlim(left=1)
            plt.ylabel(r'$x - x_{start}$')
            plt.grid()
        plt.legend()

        if self.savefig:
            plt.savefig(dest_mass + 'mass_inc_iter')
        if self.savefig_latex:
            plt.savefig(dest_latex + 'mass_inc_iter')

        if self.show_plots:
            plt.show()

    def plot_multiple_result_vectors(self,beam_model, vector_list):
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

# # DYNAMIC ANALYSIS

    def plot_dynamic_results(self, dynamic_analysis, dof_label, node_id, result_variable, init_res = None, save_suffix = '', add_fft = False):
        ''' 
        dynamic_analyis: analysis object
        dof_label: label of dof to plot
        node_id: node at whcih results to plot (starting from 0)
        result_variable: 'displacement','acceleration','reaction'
        ''' 
        dest = 'output\\'
        dof = GD.dof_lables['3D'].index(dof_label) + (node_id * GD.n_dofs_node['3D'])
        dof_for_fft = dof_label
        if result_variable == 'displacement':
            result_data = dynamic_analysis.solver.displacement[dof, :]
            if init_res:
                init_data = init_res.displacement[dof, :]
        elif result_variable == 'velocity':
            result_data = dynamic_analysis.solver.velocity[dof, :]
            if init_res:
                init_data = init_res.velocity[dof, :]
        elif result_variable == 'acceleration':
            result_data = dynamic_analysis.solver.acceleration[dof, :]
            if init_res:
                init_data = init_res.acceleration[dof, :]
        elif result_variable == 'action':
            result_data = dynamic_analysis.force[dof, :]
        elif result_variable == 'reaction':
            if dof in dynamic_analysis.structure_model.dofs_of_bc:# or dof in dynamic_analysis.structure_model.elastic_bc_dofs:
                result_data = dynamic_analysis.solver.dynamic_reaction[dof, :]
            else:
                print ('\nReplacing the selected node by the ground node for reaction results')
                node_id = 0
                dof = GD.dof_lables['3D'].index(dof_label)
                result_data = dynamic_analysis.solver.dynamic_reaction[dof, :]
            if init_res:
                init_data = init_res.dynamic_reaction[dof, :]
            dof_label = GD.direction_response_map[dof_label] 

        digits = 2
        mean = round(np.mean(result_data), digits)
        std = round(np.std(result_data), digits)
        if init_res:
            mean_init = round(np.mean(init_data), digits)
            std_init = round(np.std(init_data), digits)


        plot_title = result_variable.capitalize() + ' at node ' + str(node_id) + ' in ' + dof_label + ' direction'

        if not self.savefig or not self.savefig_latex:
            title = 'dynamic result ' + save_suffix
        else:
            title = None
        fig = plt.figure(num=title)#figsize=(5,2.2)
        ax = fig.add_subplot(111)
        plt.title(plot_title + ' Vs Time ' + save_suffix)    # set title
        ax.set_xlabel('time [s]')
        ax.set_ylabel(result_variable + ' magnitude')

        ax.hlines(mean, dynamic_analysis.array_time[0], dynamic_analysis.array_time[-1], label='mean', color = 'k')
        ax.hlines(mean + std, dynamic_analysis.array_time[0], dynamic_analysis.array_time[-1], label='mean +/- std', color = 'k', linestyle= '--')
        ax.hlines(mean - std, dynamic_analysis.array_time[0], dynamic_analysis.array_time[-1], color = 'k', linestyle= '--')
        
        label2 = None
        if init_res:
            ax.plot(dynamic_analysis.array_time, init_data, linestyle = '--', label = 'uncoupled result', color = 'tab:orange')
            label2 = 'coupled_result'

        ax.plot(dynamic_analysis.array_time, result_data, label = label2)
        
        ax.legend()

        plt.grid()
        if self.savefig:
            plt.savefig(dest)
            plt.close()

        if self.show_plots:
            plt.show()

        if add_fft:
            self.plot_fft(dof_label=dof_for_fft, dynamic_analysis=dynamic_analysis, init_dyn_res = init_res)    


    def plot_fft(self, dof_label, dynamic_analysis = None, given_series = None, sampling_freq = None, init_dyn_res = None):
        ''' 
        either give it:
            - a dynamic analysis object or 
            - directly a time series and the sample freqeuency
        
        dof_label: label of dof 
        ''' 
        fig = plt.figure(num='frequency_domain_result')#, figsize=(5,3)
        is_type = 'action '
        if dynamic_analysis:
            sampling_freq = 1/dynamic_analysis.dt
            dof = GD.dof_lables['3D'].index(dof_label)
            time_series = dynamic_analysis.solver.dynamic_reaction[dof, :]
            given_series = time_series
            is_type = 'reaction '

        label2 = None
        if init_dyn_res:
            sampling_freq_init = 1/init_dyn_res.dt
            dof = GD.dof_lables['3D'].index(dof_label)
            time_series_init = init_dyn_res.dynamic_reaction[dof, :]
            freq_half_init, series_fft_init = utilities.get_fft(time_series_init, sampling_freq_init)
            plt.plot(freq_half_init, series_fft_init, label = 'uncoupled result', linestyle = '--', color = 'tab:orange')
            label2 = 'coupled result'


        freq_half, series_fft = utilities.get_fft(given_series, sampling_freq)
        plt.plot(freq_half, series_fft, label = label2)

    
        plt.xlim(0.01,0.8)
        plt.ylabel('|Amplitude|')
        plt.xlabel('frequency [Hz]')
        plt.title(is_type + GD.direction_response_map[dof_label] + ' in the frequency domain using FFT ')
        plt.grid(True)
        plt.legend()
        plt.show()


    def plot_compare_energies(self, values_dict):

        fig = plt.figure(num='modal energy')
        norm = 1/values_dict['uncoupled']
        for i, val in enumerate(values_dict.items()):
            plt.bar(i, val[1] * norm, label=val[0])
            plt.text(i, val[1]*norm + 0.1,  str(val[1]*norm))

        plt.legend()
        plt.ylabel('sum of energy over time')
        if self.show_plots:
            plt.show()


    def set_ax_size(self, w,h, ax=None):
        """ w, h: width, height in inches """
        #if not ax: ax=plt.gca()
        l = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(w)/(r-l)
        figh = float(h)/(t-b)
        ax.figure.set_size_inches(figw, figh)