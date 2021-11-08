from inverser import RjMCMC
import matplotlib
import matplotlib.pyplot

# --------------------- load data ---------------------
f = open('/Users/nancyh/Documents/work/lab/CoFI/cofi/cofi_inverse/rjmcmc/data.txt', 'r')
lines = f.readlines()

x = []
y = []

for line in lines:
    columns = line.split()

    x.append(float(columns[0]))
    y.append(float(columns[1]))

f.close()

# ---------------------- analysis ---------------------

sigma = 3.0
n = [sigma] * len(x)

inverser = RjMCMC(x, y, n)

inverser.solve()
# print(result)

# different maximum polinomial order
order_hist = []
burnin = 100
total = 1000
orderlimit = 5
for maxorder in range(orderlimit + 1):
    # print('{:d}'.format(maxorder))
    inverser.solve(burnin, total, maxorder)
    order_hist.append(inverser.order_historgram())

# ---------------------- order hist ---------------------
# print(order_hist[5])

# ---------------------- own sampler ---------------------
sample_rate = 10
def sampler(x, y, i):
    return i % sample_rate == 0

burnin = 100
total = 1000
maxorder = 5
inverser.solve(burnin, total, maxorder, sampler)
sample_x = inverser.sample_x
sample_curves = inverser.sample_curves
results_x = inverser.results.x()
results_mean = inverser.results.mean()

fig = matplotlib.pyplot.figure()
ax = fig.add_subplot(111)

yc = 0.5
yalpha = 1.0/((1.0 - yc) * float(len(sample_curves)))
print(yalpha)
for sy in sample_curves:
    # print(sy)
    ax.plot(sample_x, sy, 
            color = str(yc),
            alpha = yalpha,
            linestyle = '-',
            linewidth = 10)

ax.plot(results_x, results_mean, 'r-')
ax.plot(x, y, 'ko')

matplotlib.pyplot.show()
