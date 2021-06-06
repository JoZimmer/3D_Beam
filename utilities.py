import numpy as np
import json

caarc_freqs = [0.231, 0.429, 0.536]
# VALUES COPIED WITH FULL MATRICIES CALCULATED
eigenmodes_target_2D_3_elems = {'y':[np.array([0.        , 0.0040305 , 0.013317  , 0.02434817]),
                                    np.array([ 0.        , -0.01444802, -0.01037197,  0.02449357]),
                                    np.array([ 0.        , -0.01819986,  0.01604842, -0.02446053])],
                                'g':[np.array([ 0.        , -0.0004894 , -0.00070823, -0.00074479]),
                                    np.array([ 0.        ,  0.00095991, -0.00161083, -0.00260447]),
                                    np.array([ 0.        , -0.0009034 , -0.00068957,  0.00432069])]}

eigenmodes_target_2D_10_elems ={'y':[np.array([0.0, -0.000223647046255273, -0.0008516138734941783, -0.0018197756020471587, -0.0030651302400258222, -0.00452698257707041, -0.006148471228742031, -0.007878364112695452, -0.009673052432583382, -0.011498680245678633, -0.01333335614230312]),
                                    np.array([ 0.        ,  0.00123514,  0.00401433,  0.00701557,  0.00911353, 0.00951618,  0.0078602 ,  0.00422764, -0.00093387, -0.00698381, -0.01333421]),
                                    np.array([ 0.        ,  0.00304246,  0.00806418,  0.01008837,  0.007016  , 0.00026276, -0.00632   , -0.00877015, -0.00526778,  0.00304816, 0.01334002])],
                                'g':[np.array([0.00000000e+00, 2.91028048e-05, 5.39127464e-05, 7.44741342e-05,9.08970510e-05, 1.03382795e-04, 1.12244221e-04, 1.17921246e-04,1.20991889e-04, 1.22179408e-04, 1.22356253e-04]),
                                    np.array([ 0.00000000e+00, -1.49126242e-04, -2.06595585e-04, -1.80905410e-04,-8.99100753e-05,  4.02818206e-05,  1.79513859e-04,  2.99658399e-04,3.81148479e-04,  4.18652321e-04,  4.24986410e-04]),
                                    np.array([ 0.00000000e+00, -3.34883542e-04, -2.77308592e-04,  3.15745412e-05,3.61060122e-04,  4.93762038e-04,  3.37172087e-04, -3.17237701e-05,-4.21134643e-04, -6.52640493e-04, -6.98021127e-04])]} 

def evaluate_residual(a_cur, a_tar):
    residual = np.linalg.norm(np.subtract(a_cur, a_tar)) / np.amax(np.absolute(a_tar))

    # print ('current: ', a_cur)
    # print ('target: ', a_tar)
    # print('residual:', residual)
    # print()
    return residual

def cm2inch(value):
    return value/2.54
    
def increasing_by(val_old, val_new):
    ''' 
    returns the increase in % from the origin = old
    ''' 
    increase = (val_new - val_old)/val_old * 100
    return round(increase,2)

def check_and_flip_sign_dict(eigenmodes_dict):
    '''
    flips the sign of y adn a deformation of modes to be positive at the first node after ground 
    dependend/coupled dofs are flip accordingly
    '''

    for idx in range(len(eigenmodes_dict['y'])):
        if eigenmodes_dict['y'][idx][1] < 0:
            eigenmodes_dict['y'][idx] = np.negative(eigenmodes_dict['y'][idx])
            try:
                eigenmodes_dict['g'][idx] = np.negative(eigenmodes_dict['g'][idx])
            except KeyError:
                pass
        try:
            if eigenmodes_dict['a'][idx][1] < 0:
                eigenmodes_dict['a'][idx] = np.negative(eigenmodes_dict['a'][idx])
        except KeyError:
            pass

    return eigenmodes_dict

def check_and_flip_sign_array(mode_shape_array):
    '''
    check_and_change_sign
    change the sign of the mode shape such that the first entry is positive
    '''
    if mode_shape_array[1] < 0:
        return mode_shape_array * -1
    elif mode_shape_array [1] > 0:
        return mode_shape_array

def analytic_function_static_disp(parameters, x, load_type = 'single'):
    l = parameters['lx_total_beam']
    EI = parameters['E_Modul'] * parameters['Iy']
    magnitude = parameters['static_load_magnitude']
    if load_type == 'single':
        #print ('  w_max soll:', magnitude*l**3/(3*EI))
        return (magnitude/EI) * (0.5 * l * (x**2) - (x**3)/6) 
    elif load_type == 'continous':
        #print ('  w_max soll:', magnitude*l**4/(8*EI))
        return -(magnitude/EI) * (-x**4/24 + l * x**3 /6 - l**2 * x**2 /4)

def analytic_eigenfrequencies(beam_model):
    # von https://me-lrt.de/eigenfrequenzen-eigenformen-beim-balken 
    parameters = beam_model.parameters
    l = parameters['lx_total_beam']
    EI = parameters['E_Modul'] * parameters['Iy']
    A = parameters['cross_section_area']
    rho = parameters['material_density']
    lambdas = [1.875, 4.694, 7.855]
    f_j = np.zeros(3)
    for i, l_i in enumerate(lambdas):
        f_j[i] = np.sqrt((l_i**4 * EI)/(l**4 * rho *A)) / (2*np.pi)
    
    return f_j

def analytic_eigenmode_shapes(beam_model):
    # von https://me-lrt.de/eigenfrequenzen-eigenformen-beim-balken 
    #parameters = beam_model.parameters
    l = beam_model.parameters['lx_total_beam']
    x = beam_model.nodal_coordinates['x0']
    lambdas = [1.875, 4.694, 7.855] # could also be computed as seen in the link

    w = []
    for j in range(3):
        zeta = lambdas[j] * x / l
        a = (np.sin(lambdas[j]) - np.sinh(lambdas[j])) / (np.cos(lambdas[j]) + np.cosh(lambdas[j])) 
        
        w_j = np.cos(zeta) - np.cosh(zeta) + a*(np.sin(zeta) -np.sinh(zeta))
        w.append(w_j)
        
        reduced_m = beam_model.m[::2,::2] 
        gen_mass = np.dot(np.dot(w_j, reduced_m), w_j)
        norm_fac = np.sqrt(gen_mass)
        w_j /= norm_fac
        is_unity = np.dot(np.dot(w_j, reduced_m), w_j)
        if round(is_unity, 4) != 1.0:
            raise Exception ('analytic mode shape normalization failed')

    return w


def save_optimized_beam_parameters(opt_beam_model, fname):
    f = open('optimized_parameters\\'+fname+'.json','w')
    
    json.dump(opt_beam_model.parameters, f)
    f.close()


def prepare_string_for_latex(string):
    if '_' in string:
        var, label = string.split('_')[0], string.split('_')[1]
        latex = r'${}$'.format(var) + r'$_{{{}}}$'.format(label)
        #return string.replace('_','')
        return latex
    else:
        return string

# # DYNAMIC ANALYSIS
def get_fft(given_series, sampling_freq):
    '''
    The function get_fft estimates the Fast Fourier transform of the given signal 
    sampling_freq = 1/dt
    '''

    signal_length=len(given_series)

    freq_half =  np.arange(0, 
                           sampling_freq/2 - sampling_freq/signal_length + sampling_freq/signal_length, 
                           sampling_freq/signal_length)

    # single sided fourier
    series_fft = np.fft.fft(given_series)
    series_fft = np.abs(series_fft[0:int(np.floor(signal_length/2))])/np.floor(signal_length/2)  
    
    max_length = len(freq_half)
    if max_length < len(series_fft):
        max_length = len(series_fft)
    
    freq_half = freq_half[:max_length-1]
    series_fft = series_fft[:max_length-1]
    
    return freq_half, series_fft