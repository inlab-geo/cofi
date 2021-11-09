#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#
# Import the libraries we will need for analysis and plotting.
#
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
# Plot the data with black crosses and the mean with a red line
#
fig = matplotlib.pyplot.figure(1)

a = matplotlib.pyplot.subplot(211)

a.plot(x, y, 'ko', xc, meancurve, 'r-')
a.set_xlim(xmin, xmax)

b = matplotlib.pyplot.subplot(212)
b.bar(xc, partlocation, xc[1] - xc[0])
b.set_xlim(xmin, xmax)

fig.savefig('ch3-order.pdf', format='PDF')

fig = matplotlib.pyplot.figure(2)

a = matplotlib.pyplot.subplot(111)
a.hist(partcount, bins=5, range=(0, 5), align='left')

fig.savefig('ch3-orderpartcount.pdf', format='PDF')

matplotlib.pyplot.show()
