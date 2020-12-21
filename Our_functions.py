#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 18:40:26 2020

@author: kreilgaard
"""

import numpy as np
from sympy import *                                    # Matlab like syntax for linear algebra and functions
import matplotlib.pyplot as plt                        # Plots and figures like you know them from Matlab
import seaborn as sns                                  # Make the plots nicer to look at
from iminuit import Minuit                             # The actual fitting tool, better than scipy's
import sys
from scipy import stats, optimize
from ExternalFunctions import Chi2Regression
from ExternalFunctions import nice_string_output, add_text_to_ax # useful functions to print fit results on figure
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


