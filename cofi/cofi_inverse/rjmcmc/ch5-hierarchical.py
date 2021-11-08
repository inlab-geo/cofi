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

lambda_min = 0.5
lambda_max = 2.0
lambda_std = 0.05

data.set_lambda_range(lambda_min, lambda_max)
data.set_lambda_std(lambda_std)


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
# Retrieve the results of the hierarchical
#
p = results.proposed()
a = results.acceptance()

print('Lambda Acceptance Rate: {:.0f}%'.format(float(a[1])/float(p[1]) * 100.0))

lh = results.lambda_history()

#
# Plot the data with black crosses and the mean with a red line
#
fig = matplotlib.pyplot.figure(1)
matplotlib.pyplot.plot(x, y, 'ko', xc, meancurve, 'r-')


fig = matplotlib.pyplot.figure(2)
matplotlib.pyplot.plot(range(len(lh)), lh)

fig = matplotlib.pyplot.figure(3)

a = matplotlib.pyplot.subplot(111)
lsamples = lh[10000:]

n, bins, patches = a.hist(lsamples, 100, range=(lambda_min, lambda_max))
a.set_title('Histogram of Lambda')
a.set_xlabel('Lambda')
a.set_ylabel('Count')

print('Lambda average: {:.0f}%'.format(sum(lsamples)/float(len(lsamples))))
fig.savefig('ch5-hierarchical.pdf', format='PDF')
matplotlib.pyplot.show()
