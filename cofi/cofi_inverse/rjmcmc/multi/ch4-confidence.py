#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#
# Import the libraries we will need for analysis and plotting.
#
import rjmcmc 
import matplotlib
import matplotlib.pyplot
from mpl_toolkits.mplot3d import axes3d, Axes3D

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
# Set our x range
#
xmin = 0.0
xmax = 10.0

#
# This is our callback function which samples the curves generated 
# during the analysis
#
sample_x = None
sample_curves = []
sample_i = 0
sample_rate = 250
def sampler_cb(x, y):
    global sample_x, sample_curves, sample_i, sample_rate

    if sample_i == 0:
        sample_x = x

    if sample_i % sample_rate == 0:
        sample_curves.append(y)

    sample_i = sample_i + 1

#
# Run a series of analyses with varying maximum allowed order
#
#
# Specify the standard deviation for the move partition 
#
pd = 1.0

#
# Run an analysis with reduced max order to only allow linear 
# segments
#
burnin = 10000
total = 50000
max_partitions = 10
max_order = 3

results = rjmcmc.regression_part1d_sampled(data, 
                                           sampler_cb,
                                           pd,
                                           burnin, 
                                           total, 
                                           max_partitions,
                                           max_order)

#
# Plot the data with black crosses, the sample curves as faint lines, and
# the mean as a red line
#
fig = matplotlib.pyplot.figure(1)
ax = fig.add_subplot(111)

yc = 0.5
yalpha = 1.0/((1.0 - yc) * float(len(sample_curves)))
for sy in sample_curves:

    ax.plot(sample_x, sy, 
            color = str(yc),
            alpha = yalpha,
            linestyle = '-',
            linewidth = 10)

ax.plot(results.x(), results.mean(), 'r-')
ax.plot(x, y, 'ko')
ax.set_xlim(xmin, xmax)

fig.savefig('ch4-confidence.pdf', format='PDF')

fig = matplotlib.pyplot.figure(2)
ax = fig.add_subplot(111)

ax.plot(results.x(), results.mean(), 'r-')
ax.plot(x, y, 'ko')
ax.plot(results.x(), results.credible_min(), 'b:')
ax.plot(results.x(), results.credible_max(), 'b:')

ax.set_xlim(xmin, xmax)

fig.savefig('ch4-confidenceintervals.pdf', format='PDF')

matplotlib.pyplot.show()



