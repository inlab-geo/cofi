#!/usr/bin/env python3
#-*- coding:utf-8 -*-

from __future__ import print_function

import math
import random

import matplotlib
import matplotlib.pyplot

#
# We define an interesting mathematical function that will require
# some higher order polynomial to fit
#
def realfunction(x):
    return math.exp(x/3.0) * math.sin(2.0*x/3.0)

#
# Set the parameters (x range etc)
#
xmin = 0.0
xmax = 10.0
npoints = 100
nsigma = 2.5

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

matplotlib.pyplot.plot(x, yn, 'ko', xs, ys, 'r-')

fig.savefig('ch0-exampledata.pdf', format='PDF')
matplotlib.pyplot.show()



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
