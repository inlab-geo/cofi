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
# Set our x range
#
xmin = 0.0
xmax = 10.0

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
# Set a range of where we think the error should be
#
lambda_min = 0.5
lambda_max = 3.0
lambda_std = 0.1

data.set_lambda_range(lambda_min, lambda_max)
data.set_lambda_std(lambda_std)

#
# Specify the standard deviation for the move partition 
#
pd = 1.0

#
# Run the default analysis
#
results = rjmcmc.regression_part1d(data, pd)

#
# Retrieve the mean curve for plotting
#
xc = results.x()
meancurve = results.mean()

#
# Retrieve the partition location and count information
#
partlocation = results.partition_location_histogram()
partcount = results.partitions()

#
# Retrieve the results of the hierarchical
#
p = results.proposed()
a = results.acceptance()
print(a, p)

print('Lambda Acceptance Rate:', float(a[4])/float(p[4]) * 100.0)

lh = results.lambda_history()

#
# Plot the data with black crosses and the mean with a red line
#
fig = matplotlib.pyplot.figure(1)

a = matplotlib.pyplot.subplot(211)

a.plot(x, y, 'ko', xc, meancurve, 'r-')
a.set_xlim(xmin, xmax)

b = matplotlib.pyplot.subplot(212)
b.bar(xc, partlocation, xc[1] - xc[0])
b.set_xlim(xmin, xmax)

fig.savefig('ch5-hierarchical.pdf', format='PDF')

fig = matplotlib.pyplot.figure(2)
matplotlib.pyplot.plot(range(len(lh)), lh)

fig = matplotlib.pyplot.figure(3)

a = matplotlib.pyplot.subplot(111)
lsamples = lh[10000:]

n, bins, patches = a.hist(lsamples, 100, range=(lambda_min, lambda_max))
a.set_title('Histogram of Lambda')
a.set_xlabel('Lambda')
a.set_ylabel('Count')

print('Lambda average:', sum(lsamples)/float(len(lsamples)))

matplotlib.pyplot.show()
