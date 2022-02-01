from cofi.samplers import ReversibleJumpMCMC
import matplotlib
import matplotlib.pyplot

# --------------------- load data ---------------------
f = open("./test_data.txt", "r")
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

inverser = ReversibleJumpMCMC(x=x, y=y, error=n)

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
yalpha = 1.0 / ((1.0 - yc) * float(len(sample_curves)))
for sy in sample_curves:
    ax.plot(sample_x, sy, color=str(yc), alpha=yalpha, linestyle="-", linewidth=10)

ax.plot(results_x, results_mean, "r-")
ax.plot(x, y, "ko")

matplotlib.pyplot.show()


# ---------------------- hierarchical ---------------------
# estimating the data noise
# lamdba: the ratio of the estimated noise to the real noise
lambda_min = 0.5
lambda_max = 2.0
lambda_std = 0.05

inverser_hie = ReversibleJumpMCMC(
    x=x,
    y=y,
    error=n,
    lambda_min=lambda_min,
    lambda_max=lambda_max,
    lambda_std=lambda_std,
)
inverser_hie.solve()

xc = inverser_hie.results.x()
meancurve = inverser_hie.results.mean()

# retrieve the results of the hierarchical
p = inverser_hie.results.proposed()
a = inverser_hie.results.acceptance()
print("Lambda Acceptance Rate: {:.0f}%".format(float(a[1]) / float(p[1]) * 100.0))
lambda_history = inverser_hie.results.lambda_history()

fig = matplotlib.pyplot.figure(1)
matplotlib.pyplot.plot(x, y, "ko", xc, meancurve, "r-")

fig = matplotlib.pyplot.figure(2)
matplotlib.pyplot.plot(range(len(lambda_history)), lambda_history)

fig = matplotlib.pyplot.figure(3)
a = matplotlib.pyplot.subplot(111)
lsamples = lambda_history[10000:]

n, bins, patches = a.hist(lsamples, 100, range=(lambda_min, lambda_max))
a.set_title("Histogram of Lambda")
a.set_xlabel("Lambda")
a.set_ylabel("Count")

print("Lambda average: {:.0f}%".format(sum(lsamples) / float(len(lsamples))))
matplotlib.pyplot.show()
