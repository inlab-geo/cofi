#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#
# Import the libraries we will need for analysis and plotting.
#

from __future__ import print_function

import rjmcmc 
import matplotlib
import matplotlib.pyplot
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.patches as mpatches
import sys

#
# Open our data file which consists of one (x, y) coordinater per line
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
# Run a series of analyses with varying maximum allowed order
#
results = []
burnin = 100
total = 1000
orderlimit = 5
for maxorder in range(orderlimit + 1):
    print('{:d}'.format(maxorder))
    results.append(rjmcmc.regression_single1d(data, burnin, total, maxorder))

colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
formats = tuple(map(lambda x: x + '-', colours))

#
# Plot the data with black crosses the curves from each of the analyses
# with a different colour
#
fig = matplotlib.pyplot.figure(1)
ax = fig.add_subplot(111)

orders = []
legendtitles = []
for result in results:

    order = result.order_histogram()
    if order == None: # The max order = 0 case will return None so 
        order = [total]

    ax.plot(result.x(), result.mean(), formats[len(orders)])

    #
    # Create the order histogram data (append zeros for orders not allowed
    # in the analyses
    #
    legendtitles.append('Max. Order %d' % len(orders))
    orders.append(order + [0] * (orderlimit + 1 - len(order)))

ax.plot(x, y, 'ko')
legendtitles.append('Data')
legend = ax.legend(legendtitles, loc='lower left')
fig.savefig('ch3-orderanalysis.pdf', format='PDF')

#
# Plot a 3D bar chart showing the progression of the order histogram
# as the analysis maximum order is increased.
#
fig = matplotlib.pyplot.figure(2)
ax = Axes3D(fig)
xs = range(orderlimit + 1)
for maxorder in xs:
    ax.bar(xs, 
           orders[maxorder], 
           zs=maxorder, 
           zdir = 'y', 
           color=colours[maxorder])

ax.set_xlabel('Order')
ax.set_ylabel('Maximum Order')
ax.set_zlabel('Count')

fig.savefig('ch3-orderanalysishist.pdf', format='PDF')

matplotlib.pyplot.show()
