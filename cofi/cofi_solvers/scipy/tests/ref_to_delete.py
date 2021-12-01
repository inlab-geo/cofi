import numpy
import matplotlib.pyplot as plt
import scipy
import scipy.optimize

# x: the arguments we want to optimise
# y: the objective function
# t: the 'x' in data prediction
# yhat: the predicted objective on t, using x


def predict(x, t):
    yhat = numpy.zeros_like(t)
    for i in range(int(numpy.shape(x)[0] / 2)):
        yhat += x[i * 2] * numpy.exp(-x[i * 2 + 1] * t)
    return yhat


# inverse textbook p28:
# > if we turn the maximization into a minimization by changing sign and
# > ignore the constant factor, then the problem becomes ...
def loglikelihood(x, t, y):
    #     yhat = numpy.zeros_like(t)
    #     for i in range(int(numpy.shape(x)[0]/2)):
    #         yhat += x[i*2]*numpy.exp(-x[i*2+1]*t)
    #     print((yhat-y).shape, numpy.transpose(yhat-y).shape)
    #     return numpy.matmul((yhat-y),numpy.transpose(yhat-y))

    yhat = predict(x, t)
    return (yhat - y) @ (yhat - y)


def gradient(x, t, y):
    #     yhat = numpy.zeros_like(t)
    #     for i in range(int(numpy.shape(x)[0]/2)):
    #         yhat += x[i*2]*numpy.exp(-x[i*2+1]*t)

    #     jac = numpy.zeros([numpy.shape(t)[0],numpy.shape(x)[0]])
    #     for i in range(int(numpy.shape(x)[0]/2)):
    #         for j in range(len(t)):
    #             jac[j,i*2]=numpy.exp(-x[i+1]*t[j])
    #             jac[j,i*2+1]=-x[i*2]*t[j]*numpy.exp(-x[i+1]*t[j])
    #     grad = numpy.matmul(numpy.transpose(jac),yhat-y)

    yhat = predict(x, t)
    jac = jacobian(x, t, y)
    grad = jac.T @ (yhat - y)
    return grad


def jacobian(x, t, y):
    jac = numpy.zeros([numpy.shape(t)[0], numpy.shape(x)[0]])
    for i in range(int(numpy.shape(x)[0] / 2)):
        for j in range(len(t)):
            jac[j, i * 2] = numpy.exp(-x[i * 2 + 1] * t[j])
            jac[j, i * 2 + 1] = -x[i * 2] * t[j] * numpy.exp(-x[i * 2 + 1] * t[j])
    return jac


def residuals(x, t, y):
    #     yhat = numpy.zeros_like(t)
    #     for i in range(int(numpy.shape(x)[0]/2)):
    #         yhat += x[i*2]*numpy.exp(-x[i*2+1]*t)

    yhat = predict(x, t)
    return yhat - y


x = numpy.array([1, 0.01])
t = numpy.linspace(0, 100, 20)
print(t.T.shape)
y = predict(x, t)
x0 = numpy.array([1.0, 0.012])
y0 = predict(x0, t)
plt.plot(t, y)
plt.plot(t, y0)


res = scipy.optimize.minimize(
    loglikelihood, x0, args=(t, y), method="Nelder-Mead", options={"disp": True}
)
res.x


res = scipy.optimize.minimize(
    loglikelihood,
    x0,
    args=(t, y),
    method="Newton-CG",
    jac=gradient,
    options={"disp": True},
)
res.x


####### 2 exponentials
x = numpy.array([1, 0.01, 2, 0.2])
t = numpy.linspace(0, 100, 100)
y = predict(x, t)
x0 = numpy.array([2, 0.001, 5, 0.1])
y0 = predict(x0, t)
plt.plot(t, y)
plt.plot(t, y0)


res = scipy.optimize.minimize(
    loglikelihood, x0, args=(t, y), method="Nelder-Mead", options={"disp": True}
)
res.x


res = scipy.optimize.minimize(
    loglikelihood, x0, args=(t, y), method="BFGS", jac=gradient, options={"disp": True}
)
res.x


######## Levenberg-Marquardt
x = numpy.array([1, 0.01, 2, 0.2, 3, 0.3])
t = numpy.linspace(0, 100, 100)
y = predict(x, t)
x0 = numpy.array([2, 0.001, 5, 0.1, 5, 1])
y0 = predict(x0, t)
plt.plot(t, y)
plt.plot(t, y0)

res = scipy.optimize.least_squares(residuals, x0, args=(t, y), method="lm")
print(res.x)
print(res.nfev, res.njev)

res = scipy.optimize.least_squares(
    residuals, x0, jac=jacobian, args=(t, y), method="lm"
)
print(res.x)
print(res.nfev, res.njev)
