#!/usr/bin/env python
# coding: utf-8

# # 3. X-ray tomography

# ### Step 1: import

# In[1]:


import numpy as np

from cofi.cofi_objective import XRayTomographyObjective
from cofi.optimisers import ScipyOptimiserSolver
from cofi.linear_reg import LRNormalEquation


# ### Step 2: define objective by specifying data path

# In[5]:


xrt_obj = XRayTomographyObjective("dataset/data_xrt.dat")


# ### Step 3: solve

# In[7]:


# scipy_solver = ScipyOptimiserSolver(xrt_obj)
# scipy_model = scipy_solver.solve()
# print(scipy_model.values())

linear_reg_solver = LRNormalEquation(xrt_obj)
linear_reg_model = linear_reg_solver.solve()
print(len(linear_reg_model.values()))
print(linear_reg_model.values())


# ### Step 4: display

# In[4]:


xrt_obj.display(scipy_model)
xrt_obj.display(linear_reg_model)


# It appears that the result given by Scipy optimizer (using its default method) is very bad... In contrast, with the information that this problem is linear, using a normal equation solver is much better in this case.

# ---
