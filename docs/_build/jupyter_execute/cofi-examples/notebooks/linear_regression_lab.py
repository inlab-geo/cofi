#!/usr/bin/env python
# coding: utf-8

# # Polynomial Linear Regression - Interactive Lab
# 
# Following the linear regression example described in [linear_regression.ipynb](linear_regression.ipynb), here let's relax and play around!

# ---
# ### Import modules and get prepared
# 
# Still, some steps are necessary in preparation for the coming interactive lab.

# In[1]:


######## Import required modules
import numpy as np
import matplotlib.pyplot as plt
from cofi import BaseProblem, InversionOptions, InversionRunner
from cofi.solvers import solvers_table

######## Set random seed (to ensure consistent results in different runs)
np.random.seed(42)

######## Define the polynomial linear regression problem
_basis_func = lambda x: np.array([x**i for i in range(4)]).T
_m_true = np.array([-6,-5,2,1])                                            # m
_sample_size = 20                                                          # N
x = np.random.choice(np.linspace(-3.5,2.5), size=_sample_size)             # x
forward_func = lambda m: (np.array([x**i for i in range(4)]).T) @ m        # m -> y_synthetic
y_observed = forward_func(_m_true) + np.random.normal(0,1,_sample_size)    # d

inv_problem = BaseProblem()
inv_problem.name = "Polynomial Regression"
inv_problem.set_dataset(x, y_observed)
inv_problem.set_forward(forward_func)
inv_problem.set_jacobian(_basis_func(x))

######## Review the basic/fixed problem setup
# inv_problem.summary()


# In[2]:


def adjust_problem(initial_model, data_misfit, regularisation, regularisation_factor):
    inv_problem.set_initial_model(initial_model)
    inv_problem.set_data_misfit(data_misfit)
    inv_problem.set_regularisation(regularisation, regularisation_factor)
    return inv_problem
    
def adjust_options(solving_method, tool, **kwargs):
    inv_options = InversionOptions()
    inv_options.set_solving_method(solving_method)
    inv_options.set_tool(tool)
    inv_options.set_params(**kwargs)
    return inv_options

def plot_from_model(model):
    _x_plot = np.linspace(-3.5, 2.5)
    _G_plot = _basis_func(_x_plot)
    _y_plot_true = _G_plot @ _m_true
    _y_plot_synth = _G_plot @ model
    plt.figure(figsize=(12,8))
    plt.plot(_x_plot, _y_plot_true, color="darkorange", label="true model")
    plt.plot(_x_plot, _y_plot_synth, color="seagreen", label="inversion result")
    plt.scatter(x, y_observed, color="lightcoral", label="original data")
    plt.title("Polynomial linear regression: comparison between true model, dataset & inversion result")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

def inversion(m0, m1, m2, m3, data_misfit, reg, reg_factor, method, tool, verbose):
    initial_model = np.array([m0, m1, m2, m3])
    inv_problem = adjust_problem(initial_model, data_misfit, reg, reg_factor)
    inv_options = adjust_options(method, tool)
    inv_options.set_params(verbose=verbose)
    inv_runner = InversionRunner(inv_problem, inv_options)
    result = inv_runner.run()
    # result.summary()
    plot_from_model(result.model)
    return result
    
# inversion(1, 1, 1, 1, "L2", "L2", 0.05, "optimisation", "scipy.optimize.minimize")


# ---
# ### Start the lab

# In[3]:


from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

m0_widget = widgets.FloatSlider(min=-10,max=10)
m1_widget = widgets.FloatSlider(min=-10,max=10)
m2_widget = widgets.FloatSlider(min=-10,max=10)
m3_widget = widgets.FloatSlider(min=-10,max=10)
reg_widget = widgets.Dropdown(value="L2",options=["L0","L1","L2"])
reg_factor_widget = widgets.FloatLogSlider(base=10,value=0.08,min=-5,max=1,step=0.2)
method_widget = widgets.ToggleButtons(options=["optimisation", "linear least square"])
tool_widget = widgets.RadioButtons(options=["scipy.optimize.minimize", "scipy.optimize.least_squares"])
verbose_widget = widgets.IntSlider(value=0,min=0,max=2,layout={"visibility":"hidden"})

def method_updated(*args):
    tool_widget.options = solvers_table[method_widget.value].keys()
method_widget.observe(method_updated, 'value')

def tool_updated(*args):
    if tool_widget.value == "scipy.linalg.lstsq":
        m0_widget.layout.visibility = "hidden"
        m1_widget.layout.visibility = "hidden"
        m2_widget.layout.visibility = "hidden"
        m3_widget.layout.visibility = "hidden"
        reg_widget.layout.visibility = "hidden"
        reg_factor_widget.layout.visibility = "hidden"
    else:
        m0_widget.layout.visibility = "visible"
        m1_widget.layout.visibility = "visible"
        m2_widget.layout.visibility = "visible"
        m3_widget.layout.visibility = "visible"
        reg_widget.layout.visibility = "visible"
        reg_factor_widget.layout.visibility = "visible"
    if tool_widget.value == "scipy.optimize.least_squares":
        verbose_widget.layout.visibility = "visible"
    else:
        verbose_widget.layout.visibility = "hidden"
tool_widget.observe(tool_updated, 'value')
        
w = interactive(inversion, {'manual': True, 'manual_name': "Run Inversion"},
                m0=m0_widget, m1=m1_widget, m2=m2_widget, m3=m3_widget,
                data_misfit=["L2"], reg=reg_widget, reg_factor=reg_factor_widget,
                method=method_widget, tool=tool_widget, verbose=verbose_widget)
display(w)


# In[5]:


inv_result = w.result
inv_result.summary()


# ---
