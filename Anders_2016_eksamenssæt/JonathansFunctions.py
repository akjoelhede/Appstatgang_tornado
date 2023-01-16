# =============================================================================
#  My functions
# =============================================================================



import warnings as warn
import sys
sys.path.append(r'D:\GitHub_D\AppStat2021\External_Functions')
import ExternalFunctions as Ef  
from iminuit import Minuit
from scipy import stats
import scipy.stats as ss

import matplotlib.pyplot as plt
import numpy as np
def Normal(x,sigma, mu, A = 1 ):
    from numpy import sqrt, exp, pi
    return A/(sigma*sqrt(2 * pi))*exp(-1/2*((x-mu)/sigma)**2)
    
plt.rcParams.update({'font.size': 14})


###
#
# 1: funktion skrevet i sympy
# 2: liste af navne af variable
# 3: værdier for hver variabel ordnet
# 4: ussikerheder for hver variabel ordnet
# 5: cov matrix
#
###

def chi2(y,f,sy):
    return np.sum((y - f)**2/sy**2)

def ophob(f,variables,values,uncertainties, cov = None, ftype = 'Sympy', verbose  = False):
    from sympy.tensor.array import derive_by_array
    from numpy import identity, array, dot, matmul
    from latex2sympy2 import latex2sympy
    from sympy import sqrt, Symbol, latex
    from sympy.abc import sigma
    
    if ftype == 'LaTeX':
        f = latex2sympy(f)


    if type(cov) == type(None):
        cov = np.diag(np.array(uncertainties)**2)


    subs_dict = dict(zip(variables, values))
    gradient = derive_by_array(f,variables).subs(subs_dict)
    
    VECTOR = array([element.evalf() for element in gradient])


    if verbose:
        print(cov)

        from sympy.printing.latex import print_latex
        print('           -- python --         ')
        print(derive_by_array(f,variables))

        print('\n         -- LaTeX  --         ')
        print_latex(derive_by_array(f,variables)) 

        print('\n         -- variables  --         ')
        print(variables)


        print('\n         -- value  --         ')
        print(f.subs(subs_dict).evalf())


        print('\n         -- Forsøg på at sammensætte  --         ')
        F = 0


        for i in range(len(variables)):
            var  = str(variables[i])
            term = derive_by_array(f,variables)[i]
            F += term**2 * Symbol('sigma_' + var)**2

        
        print(latex(sqrt(F)))

    return float(dot(VECTOR  , matmul(cov , VECTOR))**0.5)


def hist_2d_plot(X,Y, bins, range_x,range_y, xlabel, ylabel,text_loc = (0.02,0.97), aspect = None):
    plt.rcParams.update({'font.size': 20})
    
    fig, ax = plt.subplots(figsize=(14, 10))
    counts, xedges, yedges, im = ax.hist2d(X,Y, bins = bins, range=[range_x, range_y], cmin=1)
    fig.colorbar(im) # ticks=[-1, 0, 1]


    d = {'Entries': len(X),
     'Mean ' + xlabel : X.mean(),
     'Mean ' + ylabel : Y.mean(),
     'Std  ' + xlabel : X.std(ddof=1),
     'Std  ' + ylabel : Y.std(ddof=1),
     'correlation' : np.cov(X,Y)[1,0]/(np.std(X)*np.std(Y))
    }

    text = Ef.nice_string_output(d, extra_spacing=2, decimals=3)
    Ef.add_text_to_ax(*text_loc, text, ax, fontsize=15);
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout()

    return fig,ax


from numba import njit
#                                            
def ophob_numerical(f, values, uncertainties, N = 10**6, cov = None, verbose  = False, plot = False, 
    bins_2d = 50, xlabel_2d = 'fix1', ylabel_2d = 'fix2',text_loc_2d=(0.02,0.97),
    bins = 100, xlabel = 'fix3', ylabel = 'fix4', data_label = 'Monte carlo distrubution',
    save_2dhist = None, save_hist = None, hist_func = lambda x:x
    ):
    guassians = []

    if type(cov) == type(None):
        cov = np.identity(len(values))
        for i in range(len(values)):
            guassians.append(np.random.normal(values[i], uncertainties[i], N))
    else:
        theta = 0.5 * np.arctan( 2.0 * cov[1][0] / ( np.square(uncertainties[0]) - np.square(uncertainties[1]) ) )
        sigu = np.sqrt( np.abs( (((uncertainties[0]*np.cos(theta))**2) - (uncertainties[1]*np.sin(theta))**2 ) / ( (np.cos(theta))**2 - np.sin(theta)**2) ) )
        sigv = np.sqrt( np.abs( (((uncertainties[1]*np.cos(theta))**2) - (uncertainties[0]*np.sin(theta))**2 ) / ( (np.cos(theta))**2 - np.sin(theta)**2) ) )
        u = np.random.normal(0.0, sigu, N)
        v = np.random.normal(0.0, sigv, N)
        x1_all = values[0] + np.cos(theta)*u - np.sin(theta)*v
        x2_all = values[1] + np.sin(theta)*u + np.cos(theta)*v
        guassians = np.array([x1_all, x2_all])
        if plot:
            hist_2d_plot(x1_all,x2_all, bins_2d, range_x=(min(x1_all),max(x1_all)),
            range_y=(min(x2_all),max(x2_all)), xlabel= xlabel_2d, ylabel=ylabel_2d, 
            text_loc = text_loc_2d)
            if type(save_2dhist) == type('None'):
                plt.savefig(save_2dhist,dpi = 300)

    Y_gauss = f(*guassians)
    if plot:
        Y_gauss_plot = hist_func(Y_gauss)
        hist_ophob_num = Histogram_plot(Y_gauss_plot, bins, xlabel, data_label, bin_errors=True)
        hist_ophob_num[1].vlines(hist_func(np.mean(Y_gauss)),0,max(hist_ophob_num[2]),  'r', label ='mean', zorder = 10)
        plt.legend()
        if type(save_hist) == type('None'):
                plt.savefig(save_hist,dpi = 300)

    return np.mean(Y_gauss), np.std(Y_gauss)
    

def nice_print_mu_sigma(name,list):
    return name + str(list[0]) + ' +/- ' + str(list[1])


def ks_test_ullh(f, data, xmin = None, xmax =None, verbose = True):
    from sympy.abc import x
    import sympy as sp  
    from sympy import oo

    if xmin == None:
        xmin = -oo

    if xmax == None:
        xmax = oo
    
    normal_constant = sp.integrate(f,(x,xmin, xmax)).evalf()
    f *= 1/normal_constant

    if verbose == True:
        print('Is normal : ', sp.integrate(f,(x , xmin, xmax)).evalf())

    F = sp.integrate(f,(x, xmin, x))

    if verbose == True:
        sp.plot(F,f, xlim = (0,10), ylim = (0,1))

        plt.show()


    f_cdf_lambda = sp.lambdify(x,F)

    return stats.kstest(data, f_cdf_lambda)

def p_comparison(mu1,mu2,sigma1, sigma2, verbose = True):
    from scipy.stats import norm
    D_MU = np.abs(mu1-mu2) 
    D_SIGMA = np.sqrt(sigma1**2+sigma2**2)
    if verbose:
        print('p_value', norm.cdf(0,D_MU,D_SIGMA)*2)
        print('z_value', D_MU/D_SIGMA)

    return norm.cdf(0,D_MU,D_SIGMA)*2, D_MU/D_SIGMA
    
def z_value_trial_factor(pglobal,N):
    plocal = 1-np.exp(np.log(1-pglobal)/N)
    return stats.norm.ppf(plocal,0,1)

def Chauvenets_criterion(data, cutoff, pglobal = 0.01, PDF = ss.norm, verbose = 0, i = 0): # pdf needs to be object from scipy stats
    mu = np.mean(data)
    std = np.std(data, ddof = 1)
    p_cuttof = PDF.cdf(-cutoff,0,1)

    if i == 0:
        print('z_value with global p  : ', z_value_trial_factor(pglobal, len(data)))
    #mask = (PDF.cdf(data,mu,std)>p_cuttof) & (PDF.cdf(data,mu,std)< 1-p_cuttof) 
    mask = (abs(mu-data) < cutoff*std)
    
    
    if verbose> 0:
        print(np.where(~mask))
        print(data[~mask])
        print(np.min(np.vstack((PDF.cdf(data[~mask],mu,std),1-PDF.cdf(data[~mask],mu,std))),axis = 0))


        if verbose > 1:
            print(mask)
            print(data[mask])


    if len(data) == len(data[mask]):
        return data[mask]   
    return Chauvenets_criterion(data[mask], cutoff, pglobal, PDF, verbose, i +1)
    

def combning_numbers_with_std(x,sigma, axis = 0):
    from numpy import average, sqrt, sum, array
    if type(x) != array([1]):
        x = array(x)

    if type(sigma) != array([1]):
        sigma = array(sigma)

    mean = average(x,axis = axis, weights = sigma)
    std = sqrt(1/sum(1/(sigma)**2,axis = axis))
    return mean , std


def chisquarefit_histrogram(fitfunction, X, xmin, xmax, N_bins, startparameters, plot=False, verbose = False, weights=None, bound=None, priors=None):
    counts, bin_edges = np.histogram(X, bins=N_bins, range=(xmin, xmax), density=True)
    x = ((bin_edges[1:] + bin_edges[:-1])/2)[6:]       #cuts the first six bins away (empty bins) !!!!!!!!!!!!!!
    y = counts[6:]
    sy = np.sqrt(counts)[6:]

    chi2fit = Chi2Regression(fitfunction, x, y, sy, weights=weights, bound=bound, priors=priors)
    minuit_chi2 = Minuit(chi2fit, *startparameters)
    minuit_chi2.errordef = 1.0     
    minuit_chi2.migrad()   
    
    par = minuit_chi2.values[:]
    par_err = minuit_chi2.errors[:] 
    chi2_value = minuit_chi2.fval           
    N_NotEmptyBin = np.sum(y > 0)
    Ndof_value = N_NotEmptyBin - minuit_chi2.nfit
    Prob_value = stats.chi2.sf(chi2_value, Ndof_value) 
    
    if verbose == True:
        print(f"Chi2 value: {chi2_value:.1f}   Ndof = {Ndof_value:.0f}    Prob(Chi2,Ndof) = {Prob_value:5.3f}")

    if plot == True:
        x_axis = np.linspace(xmin, xmax, 1000)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        hist_trans = ax.hist(X, bins=N_bins, range=(xmin, xmax), histtype='step', label='histogram', density=True)
        ax.set(xlabel="x", ylabel="Frequency", xlim=(xmin-0.1, xmax+0.1))
        ax.plot(x_axis,fx_x3_fit(x_axis,*minuit_chi2.values))

        d = {'Chi2':     chi2_value,
             'ndf':      Ndof_value,
             'Prob':     Prob_value,
             }

        text = nice_string_output(d, extra_spacing=2, decimals=3)
        add_text_to_ax(0.62, 0.95, text, ax, fontsize=20)
    
    return chi2_value, Ndof_value, Prob_value, par, par_err




import numpy as np
from numba import njit

@njit(parallel = True)
def fx_x3_numba(x) :
    return x**3

@njit(parallel = True)
def acept_reject_jit(f, xmin,xmax, N_points, verbose = False):
    Y = f(np.linspace(xmin,xmax,N_points))
    y_min = min(Y)
    y_max = max(Y)

    X_rnd = np.random.uniform(xmin,xmax, N_points)
    Y_rnd = np.random.uniform(y_min, y_max, N_points)

    v1 = X_rnd[Y_rnd < f(X_rnd)]
    
    xx = np.linspace(xmin,xmax,N_points)

    if verbose:
        print('eff von neumann: ',len(v1)/N_points)
    return v1


def acept_reject_box_func(f, xmin,xmax, N_points, box_func, verbose = False, plot = False):
    r1 = np.random.random(N_points)
    x1 = box_func[1](r1)  # invers box_func [1]
    y1 = np.random.random(N_points)*box_func[0](x1)

    v1 = x1[y1 < f(x1)]
    xx = np.linspace(xmin,xmax,N_points)

    print('is invers?', np.sum(box_func[1](box_func[0](xx))-xx))

    if verbose:
        print('effectivity : ', len(v1)/N_points)

    if plot:
        from matplotlib.pyplot import plot, show
        plot(xx,f(xx))
        plot(xx,box_func[0](xx))
        show()
        return v1

    return v1




def fit_minuit_chi(model, x_data, y_data, sy, p0, limits = 'None', verbose = True, suppres_warning = False, weights=None, bound=None, priors=None):
    chi2reg_bkg = Ef.Chi2Regression(model, x_data, y_data, sy, weights=weights, bound=bound, priors=priors)
    
    minuit = Minuit(chi2reg_bkg, *p0)
    minuit.errordef = 1.0

    if type(limits) != type('None'):
        for limit in limits:
            minuit.limits[limit[0]] = limit[1]

    minuit.migrad() 
    
    if verbose:                                                 # Perform the actual fit
        display(minuit)
        print(r'-----    chi^2   -----')
        print(minuit.fval)
        print('\n')

        print(r'-----    p - value   -----')
        print(stats.chi2.sf(minuit.fval, len(y_data)-len(minuit.values[:])))
        print('\n')

        print(r'-----    values   -----')

        for i in range(len(minuit.values)):
            print(minuit.parameters[i], ' = ', minuit.values[i], r'\pm', minuit.errors[i])

        print(r'-----    ndof   -----')
        print(len(y_data)-len(minuit.values[:]))

    if not suppres_warning:
        if (not minuit.fmin.is_valid) :                                   # Check if the fit converged
            print("  WARNING: The ChiSquare fit DID NOT converge!!!")
    
    return minuit

def fit_minuit_ULH(model, data, bounds, p0, limits = 'None', 
            ks_test = False,
            ks_test_model = None,
            ks_test_limits = None,
            extended = True,
            verbose = True):
    ullhfit = Ef.UnbinnedLH(model, data, bound=bounds, extended=extended)
    
    minuit_ullh = Minuit(ullhfit , *p0)
    minuit_ullh.errordef = 0.5

    minuit_ullh.migrad() 


    if type(limits) != type('None'):
        for limit in limits:
            minuit_ullh.limits[limit[0]] = limit[1]

    
    if verbose:                                                 # Perform the actual fit
        display(minuit_ullh)
    

    if (not minuit_ullh.fmin.is_valid) :                                   # Check if the fit converged
        print("  WARNING: The fit DID NOT converge!!!")

    if ks_test == True:
        import sympy as sp 

        par = minuit_ullh.values[:]
        par_name = minuit_ullh.parameters[:]
        print('miniut name: ',par_name)


        name_sympy_from_minuit = [sp.symbols(name) for name in par_name]
        print('Sympy gen minuit : ',name_sympy_from_minuit)
        subs_dict = dict(zip(name_sympy_from_minuit, par))
        print(subs_dict)
        ks_test_function = ks_test_model.subs(subs_dict)
        display(ks_test_function)
        print(ks_test_ullh(ks_test_function,data,*ks_test_limits))
    return minuit_ullh





def Histogram_plot(Data, bins, xlabel, data_label,
              ylabel = None, size = (12,6), Points = False,
              density = False, cutoff = -1, color = 'k',
              xlim = None, ylim = None,
              Nstart = 0, Nend = None, bin_errors = False,
              fig_ax = None, verbose = False):
    
    if fig_ax == None:
        fig, ax = plt.subplots(figsize = size)

    else:
        if len(fig_ax) != 2:
            sys.exit('format of fig_ax is (fig,ax) ')
        fig, ax = fig_ax



    if Points:
        counts, bin_edges = np.histogram(Data, bins = bins, density= density)
        x = (bin_edges[1:][counts>cutoff] + bin_edges[:-1][counts>cutoff])/2       
        y = counts[counts>cutoff]


        if Nend != None:
            x = x[Nstart:Nend]
            y = y[Nstart:Nend]

        elif Nstart != 0:
            x = x[Nstart:]
            y = y[Nstart:]

        sy = np.sqrt(y)
        binwidth = min(np.diff(x))

        if density:
            if verbose:
                print(density)
            sy = 1/(len(Data))*np.sqrt(y*len(Data)*binwidth)
            ax.errorbar(x, y, yerr=sy, label= data_label, fmt='.'+color,  ecolor=color, elinewidth=1, capsize=1, capthick=1, zorder = 0)
            
        
        else:
            ax.errorbar(x, y, yerr=sy, label= data_label, fmt='.'+color,  ecolor=color, elinewidth=1, capsize=1, capthick=1, zorder = 0)
    
    elif cutoff == -1:
        counts, bin_edges, art = ax.hist(Data, bins = bins, density = density, label = data_label, color = color)
        if bin_errors:
            x = (bin_edges[1:] + bin_edges[:-1])/2       
            y = counts

            if density:
                print('density: ', density)
                sy = 1/(len(Data))*np.sqrt(y*len(Data)*binwidth)
            else:
                sy = np.sqrt(y)

            ax.errorbar(x,y,yerr = sy, fmt = '.b')


    else:
        print(Points, cutoff)
        sys.exit(' Not implemented yet ')


    ax.set_xlabel(xlabel)

    if ylabel != None:
        ax.set_ylabel(ylabel)

    elif density:
        ax.set_ylabel('Frequncy denisty')

    elif not density:
        ax.set_ylabel('Frequncy')


    if xlim != None:
        ax.set_xlim(xlim)


    if ylim != None:
        ax.set_ylim(ylim)



    if Points:
        return fig, ax, x, y, sy, binwidth



    else:
        return fig, ax, counts, bin_edges, art 
  

def Plot_points(x,y, xlabel, ylabel,data_label,
              yerr = None, xerr = None, size = (12,6),
              color = 'k', style = '.',
              xlim = None, ylim = None,
              fig_ax = None):
    
    if fig_ax == None:
        fig, ax = plt.subplots(figsize = size)

    else:
        if len(fig_ax) != 2:
            sys.exit('format of fig_ax is (fig,ax) ')
        fig, ax = fig_ax

    ax.errorbar(x,y, yerr, xerr, label = data_label, fmt = style + color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if xlim != None:
        ax.set_xlim(xlim)

    if ylim != None:
        ax.set_ylim(ylim)

    plt.rcParams.update({'font.size': 14})


    return fig, ax


def line_plot(f,xmin, xmax,
            style, label, 
            xlim = None, ylim = None,
            color = 'r',  N = 10000, 
            fig_ax = None, z = 3):    

    if fig_ax == None:
        fig, ax = plt.subplots(figsize = (12,6))


    else:
        if len(fig_ax) != 2:
            print(fig_ax)
            sys.exit('format of fig_ax is (fig,ax) ')
        fig, ax = fig_ax

    x = np.linspace(xmin,xmax,N)
    ax.plot(x,f(x), style, label = label, color = color, zorder= z)

    if xlim != None:
        ax.set_xlim(xlim)


    if ylim != None:
        ax.set_ylim(ylim)
    plt.rcParams.update({'font.size': 14})






def plot_insert(fig, ax, plot_type_list, plot_input_list ,position, xlim, ylim, 
                color = 'k',
                order = 1, 
                indicate_inset = True, edgecolor = 'k'):
    

    inset_ax = ax.inset_axes(position)
    inset_ax.set_xlim(xlim)
    inset_ax.set_ylim(ylim)


    plot_type_allowed = [Histogram_plot,line_plot]
    for i in range(len(plot_type_list)):
        for P_type in plot_type_allowed:
            if plot_type_list[i] == P_type:
                plot_type_list[i](**plot_input_list[i],fig_ax = (fig, inset_ax))


    if indicate_inset:
        ax.indicate_inset_zoom(inset_ax, edgecolor=edgecolor, zorder = 10)


def weightedmean(x,sigma):
    mean = sum(x/sigma**2)/sum(sigma**-2)
    uncertainty = np.sqrt(1/sum(sigma**-2))
    return mean, uncertainty

def print_on_plot(ax, minuit, Ndof_value,x,y,decimals=3,extra_spacing=2, fontsize=15 ):
    
    
    par = minuit.values[:]
    par_name = minuit.parameters[:]

    par_err = minuit.errors[:]
    d = {}
    
    if minuit.errordef == 1:
        d = {
         'Chi2':     minuit.fval,
        'ndf':      Ndof_value,
        'Prob':     stats.chi2.sf(minuit.fval, Ndof_value),
        }
    
    for i in range(len(par)):
            d[f'{par_name[i]}'] = [par[i], par_err[i]]
    text = Ef.nice_string_output(d, extra_spacing=extra_spacing, decimals=decimals)
    Ef.add_text_to_ax(x,y, text, ax, fontsize=fontsize)


    
def fitting_straight_line(x,y,sy, plot = False, data_label = None, xlabel = 'FIX', ylabel = 'FIX',weights=None, bound=None, priors=None):
    def line(x,a):
        return a * np.ones_like(x)

    chi2fit = Ef.Chi2Regression(line, x, y, sy,weights=weights, bound=bound, priors=priors)
    minuit_chi2 = Minuit(chi2fit, a = np.mean(y))
    minuit_chi2.errordef = 1.0     
    display(minuit_chi2.migrad())

    if plot:
        fig, ax = plt.subplots(figsize = (12,6))
        xx = np.linspace(min(x),max(x),1000)
        ax.errorbar(x, y,sy, fmt = 'k.', label = data_label)
        ax.plot(xx, line(xx,*minuit_chi2.values), label = 'Fitted Constant')

        print_on_plot(ax, 
        minuit_chi2,len(x)-len(minuit_chi2.values),
        0.6,0.25, decimals=5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)


        plt.legend()
    return *minuit_chi2.values[:] ,  *minuit_chi2.errors[:],minuit_chi2.fval, stats.chi2.sf(minuit_chi2.fval, len(y)-1)




def tot_fit_function_hist(Data, bins, xlabel, data_label,
                model, p0, xmin_line, xmax_line, label_line, 
              ylabel = None, size = (12,6), Points = True,  # Histogram 
              density = False, cutoff = 0, color_hist = 'k',
              Nstart = 0, Nend = None, bin_errors = False,
              fig_ax = None,

            limits = 'None', # fit
            suppres_warning = False, 
            #line_plot
            style_line = '-', 
            color_line = 'g',  N_line = 10000, 
            xlim_line = None, ylim_line = None,
            decimals=3, extra_spacing=2, fontsize=15,
            print_loc = (1 ,1 ) , legend_loc = (0,0),
            verbose = True, #### ULLH 
            ULLH = False,
            ULLh_extended = True,
            ULLH_model = None, ULLH_bounds = None,
            label_line_ULLH = 'fix',
            color_line_ullh = 'b', print_loc_ullh = (0,1),
            Ullh_p0 = None,
            ks_test = False,
            ks_test_model = None,
            ks_test_limits = None, priors = None
              ):
    
    if verbose:
        print('cutoff : ', cutoff)

    Histogram =  Histogram_plot(
    Data, bins, 
    xlabel, data_label, 
    ylabel = ylabel, size = size, Points = Points,
    density = density, cutoff = cutoff, color = color_hist,
    Nstart = Nstart, Nend = Nend, bin_errors = bin_errors,
    fig_ax = fig_ax)

    chi_2_fit =  fit_minuit_chi(model, Histogram[2], 
            Histogram[3],  Histogram[4], p0,
            limits = limits, verbose = verbose, suppres_warning = suppres_warning,priors=priors)


    line_plot(lambda x : model(x, *chi_2_fit.values), 
            xmin_line, xmax_line, style_line, label_line,
            xlim = xlim_line, ylim = ylim_line, N = N_line, 
            fig_ax=Histogram[:2], color = color_line)

    print_on_plot(Histogram[1], chi_2_fit , 
        len(Histogram[2])-len(chi_2_fit.values), *print_loc, 
        decimals=decimals,extra_spacing=extra_spacing, fontsize=fontsize)

    if ULLH == True:
        if Ullh_p0 != None:
            p0 = Ullh_p0
        ULLh_fit_minuit =  fit_minuit_ULH(ULLH_model, Data, ULLH_bounds, 
        p0,  limits = limits, verbose = verbose, 
        extended = ULLh_extended,
        ks_test = ks_test,
        ks_test_model = ks_test_model,
        ks_test_limits = ks_test_limits  )
        
        line_plot(lambda x : ULLH_model(x, *ULLh_fit_minuit.values)*Histogram[5], 
            xmin_line, xmax_line, style_line, label_line_ULLH,
            xlim = xlim_line, ylim = ylim_line, N = N_line, 
            fig_ax=Histogram[:2], color = color_line_ullh)
        
        print_on_plot(Histogram[1], ULLh_fit_minuit , 
        len(Histogram[2])-len(ULLh_fit_minuit.values), *print_loc_ullh, 
        decimals=decimals,extra_spacing=extra_spacing, fontsize=fontsize)

    Histogram[1].legend(loc =legend_loc)

    if ULLH == True:
        return fig_ax, Histogram, chi_2_fit,ULLh_fit_minuit

    return fig_ax, Histogram, chi_2_fit




def tot_fit_function_points(x,y,sy, xlabel, ylabel, data_label,
                model, p0, xmin_line, xmax_line, label_line,
                size = (12,6), xerr = None,
              density = False, cutoff = 0, color_points = 'k', 
              style_points = '.',
              Nstart = 0, Nend = None, bin_errors = False,
              fig_ax = None,

            limits = 'None', # fit
            suppres_warning = False, 
            #line_plot
            style_line = '-', 
            color_line = 'g',  N_line = 10000, 
            xlim_line = None, ylim_line = None,
            decimals=3, extra_spacing=2, fontsize=15,
            print_loc = (1 ,1 ) , legend_loc = (0,0),
            verbose = True, #### ULLH 
            ULLH = False,
            ULLH_model = None, ULLH_bounds = None,
            label_line_ULLH = 'fix',
            color_line_ullh = 'b', print_loc_ullh = (0,1),
            ks_test = False,
            ks_test_model = None,
            ks_test_limits = None
              ):
    
    if verbose:
        print('cutoff : ', cutoff)

    plot_points =  Plot_points(x,y, xlabel, ylabel,data_label,
              yerr = sy, xerr = None, size = size,
              color = color_points, style = style_points,
              )

    chi_2_fit =  fit_minuit_chi(model, x,y,sy, p0,
            limits = limits, verbose = verbose, suppres_warning = suppres_warning)


    line_plot(lambda x : model(x, *chi_2_fit.values), 
            xmin_line, xmax_line, style_line, label_line,
            xlim = xlim_line, ylim = ylim_line, N = N_line, 
            fig_ax=plot_points, color = color_line)

    print_on_plot(plot_points[1], chi_2_fit , 
        len(x)-len(chi_2_fit.values), *print_loc, 
        decimals=decimals,extra_spacing=extra_spacing, fontsize=fontsize)


    plot_points[1].legend(loc =legend_loc)

    return plot_points, chi_2_fit, stats.chi2.sf(chi_2_fit.fval, len(x)-len(chi_2_fit.values)),


def peak_significance_calculator(N_peak, sigma_N_peak, sigma, range_size, verbose = True):
    trial_factor = range_size/sigma
    p_value = stats.norm.cdf(0,loc = N_peak, scale = sigma_N_peak)
    if verbose :
        print('p_value' , stats.norm.cdf(0,loc = N_peak/sigma_N_peak, scale = 1))
        print('p_value' , p_value)
    return 1-(1- p_value)**trial_factor


def transformation_method(pdf,inv,bins, 
bounds = (0,1), bounds_uniform = (0,1), N= 10**4,
y_func = lambda x: x, x_func= lambda x: x,
xlim = None
):
    x_linspace = np.linspace(*bounds,N)

    rnd_uni = np.random.uniform(*bounds_uniform, N)

    rnd_inv = inv(rnd_uni)

    Histogram = Histogram_plot(x_func(rnd_inv), bins, 'Number', 'Generated data', density=True)
    
    
    Histogram[1].plot(x_func(x_linspace), y_func(pdf(x_linspace)), 'r--')
    plt.legend()
    plt.xlim(xlim)
    return Histogram, rnd_inv



def calc_ROC(hist1, hist2) :

    # First we extract the entries (y values) and the edges of the histograms:
    # Note how the "_" is simply used for the rest of what e.g. "hist1" returns (not really of our interest)
    y_sig, x_sig_edges, _ = hist1 
    y_bkg, x_bkg_edges, _ = hist2
    
    # Check that the two histograms have the same x edges:
    if np.array_equal(x_sig_edges, x_bkg_edges) :
        
        # Extract the center positions (x values) of the bins (both signal or background works - equal binning)
        x_centers = 0.5*(x_sig_edges[1:] + x_sig_edges[:-1])
        
        # Calculate the integral (sum) of the signal and background:
        integral_sig = y_sig.sum()
        integral_bkg = y_bkg.sum()
    
        # Initialize empty arrays for the True Positive Rate (TPR) and the False Positive Rate (FPR):
        TPR = np.zeros_like(y_sig) # True positive rate (sensitivity)
        FPR = np.zeros_like(y_sig) # False positive rate ()
        
        # Loop over all bins (x_centers) of the histograms and calculate TN, FP, FN, TP, FPR, and TPR for each bin:
        for i, x in enumerate(x_centers): 
            
            # The cut mask
            cut = (x_centers < x)
            
            # True positive
            TP = np.sum(y_sig[~cut]) / integral_sig    # True positives
            FN = np.sum(y_sig[cut]) / integral_sig     # False negatives
            TPR[i] = TP / (TP + FN)                    # True positive rate
            
            # True negative
            TN = np.sum(y_bkg[cut]) / integral_bkg      # True negatives (background)
            FP = np.sum(y_bkg[~cut]) / integral_bkg     # False positives
            FPR[i] = FP / (FP + TN)                     # False positive rate            
            
        return FPR, TPR

def delta_std(sigma,N):
    return sigma/(np.sqrt(2*(N-1)))