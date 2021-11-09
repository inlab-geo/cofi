#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#
# Import the libraries we will need for analysis and plotting.
#
import rjmcmc 
import matplotlib
import matplotlib.pyplot

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

fig = matplotlib.pyplot.figure()

matplotlib.pyplot.plot(x, y, 'k+')
matplotlib.pyplot.show()

fig.savefig('ch1-loading.pdf', format='PDF')
