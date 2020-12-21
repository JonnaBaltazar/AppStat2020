#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 18:40:26 2020

@author: Jonna
"""

import numpy as np
from sympy import * 
from sympy import sympify                                   # Matlab like syntax for linear algebra and functions
import matplotlib.pyplot as plt                        # Plots and figures like you know them from Matlab
import seaborn as sns                                  # Make the plots nicer to look at
from iminuit import Minuit                             # The actual fitting tool, better than scipy's
import sys
from scipy import stats, optimize
from ExternalFunctions_Troels import Chi2Regression, lprint
from ExternalFunctions_Troels import nice_string_output, add_text_to_ax # useful functions to print fit results on figure
from IPython.display import display








##### WEIGHTED AVERAGE #####
def weighted_avg(val, err, plot=False, title=None):
    """
    INPUT:
    val = values, array_like
    err = erros, array_like
    plot = option to plot or not
    
    """
    
    # Calculate the avg according to Barlow (4.6)
    avg = np.sum( (val / err**2) / np.sum( 1 / err**2 ) )
    
    # Calculate the error
    avg_sig = np.sqrt( 1 / np.sum(1 / err**2) ) 
    
    # Find degrees of freedom (-1 )
    N_dof = len(val) - 1
    
    # Calculate chi_square
    chi2 = np.sum( (val - avg)**2 / err**2 )
    
    # Calculate p-value (the integral of the chi2-distribution from chi2 to infinity)
    p = stats.chi2.sf(chi2, N_dof)
    
    # Option to plot the fitted line
    if plot:
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12,6))
        
        # X values are measurement number
        x = np.arange(len(val))+1
        
        # Plot values and errorbars
        ax.scatter(x, val)
        ax.errorbar(x, val, err, fmt='ro', ecolor='k', elinewidth=1, capsize=2, capthick=1)
        
        #Plot the weighted average line
        ax.hlines(avg, 0, len(val)+0.5, colors = 'r', linestyle = 'dashed')
        
        # Nice text
        d = {'mu':   avg,
             'sigma_mu': avg_sig,
             'Chi2':     chi2,
             'ndf':      N_dof,
             'Prob':     p,
            }

        text = nice_string_output(d, extra_spacing=2, decimals=3)
        add_text_to_ax(0.02, 0.95, text, ax, fontsize=14)
        ax.set_title(title, fontsize=18)
        fig.tight_layout()

    return avg, avg_sig, chi2, p



##### ERROR PROPAGATION #####
# Define a function that will generate the function that evaluates the value and error and contributions
def value_error_contribution_func_gen(expr, print_lvl, variables):
    """
    expr = takes in a math expression in a string of type 'a+b'
    var = takes in a tuple of variables strings, fx ('a', 'b') 
    """
    # Convert expression into a sympy expression
    expr = sympify(expr)
    
    # Define sympy symbols for the parameters (the tuple variables) and the standard deviations
    var_symbols = symbols(variables)
    err_symbols = symbols( tuple("sigma_" + k for k in variables) )
    
    # Find expressions for each contributions
    contributions = [expr.diff(var) ** 2 * err**2 for var, err in zip(var_symbols, err_symbols)]
    
    # Convert contributions to numerical functions
    f_contributions = [ lambdify(var_symbols + err_symbols, expression) for expression in contributions ]

    # Find the error propagation expression to be evaluated, and display
    expr_sig = sqrt( sum(contributions) )
    
    if print_lvl == 1:
        display(expr_sig)
    
    # Convert the expression for the value and the error into numerical functions
    f_val = lambdify(var_symbols, expr)
    f_err = lambdify(var_symbols + err_symbols, expr_sig)
    
    def func(**kwargs):
        """
        Define a function that will take in keywordarguments **kwargs which is a dictionary of type: 
        {'a':(1,0.1), 'b':(2,0.3)}. Kwargs.values calls the two tuples as one list [(1,0.1),(2,0.3)].
        From there an array of variables and an array of errors can be extracted and the numerical
        functions found above can be used.
        
        """
        # Create tuple of values of variables
        v = tuple(v[0] for v in kwargs.values())
        
        # Create tuple of errors of variables
        s = tuple(v[1] for v in kwargs.values())
        
        # Calculate value and error
        value, error = f_val(*v), f_err(*v, *s)

        # Calculate contribution from each variable
        contr_list = [ function(*v,*s) for function in f_contributions ]
        
        #Return value and analytical error
        return value, error, contr_list
    
    # Return the main function that we set out to generate
    return func


# Define function that gets variables from **kwargs and uses the function above to return value and error
def val_err_contr(expr, print_lvl = 1, **kwargs):
    """
    INPUT:
    expr = takes in a math expression in a string of type 'a+b'
    **kwargs = variable names = (value, error) of type a=(3, 0.3)
    
    Note that if the relation depends on constant, type those in as variables with sigma = zero.
    
    
    OUTPUT:
    value = integer
    error = integer
    contributions = array_like with contributions from each variable in the same order as in the input
    """
    
    return value_error_contribution_func_gen(expr, print_lvl, tuple(kwargs))(**kwargs)



# We define a function that can return C. It returns multiple values, and the positive real value is the 
# one that needs to be chosen

def find_C(expr, variables):
    
    """
    INPUT:
    expr = Math expression in a string, ex. 'a+b'
    
    Only works for x to the power of a real number, ex. it does NOT WORK for x**b.  
    """

    # Convert expression into a sympy expression
    expr = sympify(expr)
    
    # Define sympy symbols for the parameters:
    var_symbols = symbols(variables)
    
    # We define the symbols that are always needed:
    x, f, F, C = symbols("x, f, F, C")
    
    # We integrate the function, and set C in the place of x: 
    F = integrate(expr, (x, 0, C))
    
    # We solve for F(C) = 1, which  
    C_1 = solve(F - 1, C)
    
    # We print the function along with the Integrated function : 
    lprint(latex(Eq(symbols('f'), expr)))
    lprint(latex(Eq(symbols('F'), F)))
    
    # We loop over all the possible C-values: 
    print('The possible C values are:')
    for i in range(len(C_1)):
        print(i)
        lprint(latex(Eq(symbols('C'), C_1[i])))
    
    # We return all of the C's. Find the real positive number, and index for that. Often C_1[0]. 
    return C_1


#def val_err_contr(expr, **kwargs):
#    return findC(expr, tuple(kwargs))**kwargs



def monte_c(expr, N_points, N_bins, xmini, xmaxi, ymini, ymaxi, plot = False, title=False):
    
    """
    expr = defined function 
    N_points = number of random numbers
    N_bins = number of bins
    
    Define the box:
    
    xmin = xmin for the box, 0
    xmax = xmax for the box, C found from the find_C function 
    ymin = ymin for the box, 0
    ymax = ymax for the box, expr as a function of C, f(C)   
    
    """
    
    xmax, xmin, ymax, ymin = float(xmaxi), float(xmini), float(ymaxi), float(ymini)
    
    # We start by creating the random numbers: 
    
    #xmin, xmax, ymin, ymax = int(xmini), int(xmaxi), int(ymini), int(ymaxi)    
    
    
    r = np.random
    r.seed(42)
    
    # We set the N_try to zero, which we loop over: 
    N_try = 0
    
    # We create an empty array for which we add the accepted numbers: 
    
    x_accepted = np.zeros(N_points)

    for i in range(N_points):
    
        while True: #While true loops means run until break:
        
            # Count the number of tries, to get efficiency/integral
            N_try += 1                    
        
            # Range that f(x) is defined/wanted in:
            x_test = r.uniform(xmin, xmax)  
        
            # Upper bound for function values:
            y_test = r.uniform(ymin, ymax)
        
            if (y_test < expr(x_test)) :
                
                break
            
        x_accepted[i] = x_test
        
    # Efficiency
    eff = N_points / N_try  
    
    N_bins = 100

    # Error on efficiency (binomial!)
    eff_error = np.sqrt(eff * (1-eff) / N_try) 

    # Integral
    integral =  eff * (xmax-xmin) * (ymax-ymin)

    # Error on integral
    integral_err = eff_error * (xmax-xmin) * (ymax-ymin) 
    
    
    if plot : 
        
        # We need to scale the expression to fit to the number of points: 
        k = (xmax - xmin) / N_bins
        N = N_points * k

        # We plot over a simple linspace, thereby the function is not fitted
        x_axis = np.linspace(xmin, xmax, 1000)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # We plot both the histogram of the accepted values, and the scaled non-fitted function: 
        ax.hist(x_accepted, bins = N_bins, range=(xmin, xmax), histtype='step', label='Histogram' )
        ax.set(xlabel="X", ylabel="Frequency", xlim=(xmin-0.1, xmax+0.1));
        ax.plot(x_axis, N * expr(x_axis) , 'r-', label='Function (not fitted)')

        # Define figure text
        d = {'Entries': len(x_accepted),
             'Mean': x_accepted.mean(),
             'Std':  x_accepted.std(ddof=1),
            }

        # Plot figure text
        text = nice_string_output(d, extra_spacing=2, decimals=3)
        add_text_to_ax(0.05, 0.75, text, ax, fontsize=16)

        # Add legend
        ax.legend(loc='best')
        fig.tight_layout()

        """
        OUTPUT:
        Efficiency
        Error Efficiency
        
        Value under integral
        Error integral
        """
    return eff, eff_error, integral, integral_err





