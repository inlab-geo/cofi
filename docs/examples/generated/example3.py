"""
"This" is my example-script
===========================

This example doesn't do much, it just makes a simple plot
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.array([1,2,3,4,5,6,7,8])
y = x ** 2 - 1

plt.scatter(x,y,color="purple")

