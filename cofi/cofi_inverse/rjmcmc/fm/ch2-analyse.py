#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#
# Import the libraries we will need for analysis and plotting.
#

from __future__ import print_function

import sys
import rjmcmc 
import matplotlib
import matplotlib.pyplot

xmin = 0.0
xmax = 10.0
pd = 1.0

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
# We set an estimate of the noise parameter. This will affect the tightness
# of the fit.
#
sigma = 5.0

#
# Local parameters are the values in each partition, here since we are
# doing a simple regression there is one value in each partition. We 
# specify a tuple of (min, max, std. deviation) for each parameter.
#
local_parameters = [(-50.0, 50.0, 2.0)]

#
# Global parameters are parameters that are the same in all partitions,
# an example is a dc bias that is constant across the domain. In this 
# example we have no global parameters. They are specified in the 
# same manner as the local parameters.
#
global_parameters = None

def my_loglikelihood(globalvalues, boundaries, localvalues):

    global x, y, sigma

    #
    # The log(likelihood) callback function needs to return the error
    # between the proposed model and the data. In this case we are 
    # doing a simple regression (there is no actual forward model 
    # transformation). The parameters to this function are a list
    # of the global values, a sorted list of the boundary positions 
    # including the start and end position, and the values in each
    # region (so len(boundaries) == len(localvalues) + 1).
    #

    #
    # phi is our error accumulator
    #
    phi = 0.0

    #
    # Loop through our data points
    #
    for (xi, yi) in zip(x, y):
        
        if (xi < boundaries[0]) or (xi > boundaries[len(boundaries) - 1]):
            #
            # if the point is outside the range then ignore it, this shouldn't
            # happen
            #
            continue

        #
        # Find which cell this coordinate is in
        #
        ci = 0
        while (ci < len(localvalues)) and (xi > boundaries[ci + 1]):
            ci = ci + 1

        if ci == len(localvalues):
            print('error: failed to find cell')
            print('  ', boundaries, xi)
            sys.exit(-1)

        #
        # The localvalues parameter is a 2d array since we can have more than
        # one value per cell. In this example, there is only one value per
        # cell.
        #
        ym = localvalues[ci][0]

        #
        # Accumulate the square of the error
        #
        dy = yi - ym
        phi = phi + (dy * dy)
        
    return phi/(2.0 * sigma * sigma)

#
# Run the default analysis
#
results = rjmcmc.forwardmodel_part1d(local_parameters,
                                     global_parameters,
                                     my_loglikelihood,
                                     xmin,
                                     xmax,
                                     pd)
if (results == None):
    print('error: failed to run forward model')
    sys.exit(-1)

#
# Retrieve the mean curve for plotting
#
xc = results.x()
meancurve = results.mean()

p = results.proposed()
a = results.acceptance()

print(p)
print(a)

misfit = results.misfit()
partitions = results.partitions()

partlocation = results.partition_location_histogram()

#
# Plot the data with black crosses and the mean with a red line
#
fig = matplotlib.pyplot.figure()

a = matplotlib.pyplot.subplot(211)

a.plot(x, y, 'k+', xc, meancurve, 'r-')
a.set_xlim(xmin, xmax)

b = matplotlib.pyplot.subplot(212)
b.bar(xc, partlocation, xc[1] - xc[0])
b.set_xlim(xmin, xmax)

fig.savefig('ch2-analyse.pdf', format='PDF')

matplotlib.pyplot.show()
