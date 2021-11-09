#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import math
import random

import matplotlib
import matplotlib.pyplot

#
# We define an simple function with 2 discontinuities
#
def realfunction(x):
    if x < 3.3:
        return 20.0
    elif x < 6.6:
        return 30.0
    else:
        return 10.0

#
# Set the parameters (x range etc)
#
xmin = 0.0
xmax = 10.0
npoints = 100
nsigma = 5.0


#
# Create a set of randomly spaced but fairly evenly distributed x 
# coordinates to sample the function
#
dx = (xmax - xmin)/float(2*npoints)
x = [dx + 2.0*dx*float(a) + random.uniform(-dx, dx) for a in range(npoints)]

#
# Create a higher sampled x vector for the smooth real curve
#
nsmoothpoints = 100
xs = [float(x)/float(nsmoothpoints - 1) * (xmax - xmin) + xmin for x in range(nsmoothpoints)]

#
# Determine the real values at those x-coordinates
#
y = list(map(realfunction, x))
ys = list(map(realfunction, xs))

#
# Add some noise
#
yn = [y + random.normalvariate(0.0, nsigma) for y in y]

#
# Plot the noisy data with black crosses and the real function with 
# a red line
#
fig = matplotlib.pyplot.figure()

matplotlib.pyplot.plot(x, yn, 'k+', xs, ys, 'r-')
matplotlib.pyplot.show()

fig.savefig('ch0-exampledata.pdf', format='PDF')

#
# Save the data in a space separated file
#
f = open('data.txt', 'w')
for xc, yc in zip(x, yn):
    f.write('%f %f\n' % (xc, yc))
f.close()

f = open('data_real.txt', 'w')
for xc, yc in zip(x, y):
    f.write('%f %f\n' % (xc, yc))
f.close()
