#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#
# Import the libraries we will need for analysis and plotting.
#

from __future__ import print_function

import rjmcmc 
import matplotlib
import matplotlib.pyplot

#
# Open our data file which consists of one (x, y) coordinate per line
# separated by whitespace
#
f = open('data.txt', 'r')
lines = f.readlines()

x = []
y = []

for line in lines:
    columns = line.split()

    x.append(float(columns[0]))
    y.append(float(columns[1]))

f.close()

#
# Estimate our error standard deviation
#
sigma = 3.0
n = [sigma] * len(x)

#
# Create the rjmcmc dataset
#
data = rjmcmc.dataset1d(x, y, n)

#
# Run the default analysis
#
results = rjmcmc.regression_single1d(data)

#
# Retrieve the mean curve for plotting
#
xc = results.x()
meancurve = results.mean()

#
# Plot the data with black crosses and the mean with a red line
#
fig = matplotlib.pyplot.figure()
matplotlib.pyplot.plot(x, y, 'ko', xc, meancurve, 'r-')
fig.savefig('ch2-analyse.pdf', format='PDF')
matplotlib.pyplot.show()
