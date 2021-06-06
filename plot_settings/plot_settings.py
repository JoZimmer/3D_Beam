import matplotlib.pyplot as plt 

def cm2inch(value):
    return value/2.54
# global PLOT PARAMS
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
# paper size for a4 landscape
# width = cm2inch(29.7)
# height = cm2inch(21)
#% width: 14.8cm or 7.3cm  or 4.8cm recommended for figures in separated rows

def get_params(w=16, h=10):
    ''' 
    w: width of savefig in [cm]
    h: height of savefig in [cm]
    
    paper size A4 landscape: 29.7, 21 [cm]
    recommended for figures in seperated rows:
        width: 14.8 or 7.3 or 4.8 [cm]
    '''
    width = cm2inch(w) # 16
    height = cm2inch(h) # 11
    params = {
            # FIGURE
            # ***********************
            'text.usetex': True,
            'font.size': 6,
            #'font.family': 'sans-serif', # lmodern -> not supported when usetex?!
            #'font.sans-serif':['Helvetica'],
            #'text.latex.unicode': True,
            'figure.titlesize': 8,
            'figure.figsize': (width, height),
            'figure.dpi': 300, # for showing!!
            #'figure.constrained_layout.use': True,
            # SUBPLOTS
            # ************************
            # USE with the suplot_tool() to check which settings work the best
            'figure.subplot.left': 0.1,
            'figure.subplot.bottom': 0.15,
            'figure.subplot.right': 0.9,
            'figure.subplot.top': 0.8,
            'figure.subplot.wspace': 0.20,
            'figure.subplot.hspace': 0.30,
            # LEGEND
            # ***********************
            'legend.fontsize': 6,
            # AXES
            # ***********************
            'axes.titlesize': 8,
            'axes.titlepad': 6,
            'axes.labelsize': 6,
            'axes.labelpad': 4,
            'axes.xmargin': 0.1,
            'axes.ymargin': 0.1,
            # FORMATTER
            # ***********************
            'axes.formatter.limits':(-3,3), #limit der tausender ab wann sie in potenzen dargestellt werden 
            #'axes.formatter.min_exponent': 4,
            # TICKS
            # ***********************
            'xtick.labelsize': 6,
            'ytick.labelsize': 6,
            'ytick.minor.visible': True,
            'xtick.minor.visible': True,
            # GRIDS
            # ***********************
            'axes.grid': False, # NOTE: grid = True and which = both or major doesn't show the majors, for this gird must be False then which works correctly
            'axes.grid.which': 'major',#both
            'grid.linestyle': '-',
            'grid.linewidth': 0.25,
            'grid.alpha': 0.5,
            # LINES
            # ***********************
            'lines.linewidth': 0.7,
            'lines.markersize': 1,
            'lines.markerfacecolor': 'darkgrey',
            # CONTOUR PLOTS
            # **********************
            #'contour.linewidth': 0.4, removed?!
            # TEXT
            # ***********************
            'mathtext.default': 'regular',
            # SAVING
            # ***********************
            'savefig.dpi': 300,
            'savefig.format': 'pdf',
            'savefig.bbox': 'tight',
            # ERRORBARS
            # ***********************
            'errorbar.capsize': 2.0,

        }
    return params