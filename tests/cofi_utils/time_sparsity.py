import time
import numpy
import cofi

model_size = (10000,)
test_model = numpy.random.random(model_size)

time0 = time.time()

my_reg = cofi.utils.QuadraticReg(model_shape=model_size)
time1 = time.time()

my_reg(test_model)
time2 = time.time()

my_reg.gradient(test_model)
time3 = time.time()

my_reg.hessian(test_model)
time4 = time.time()

print(time1-time0, time2-time1, time3-time2, time4-time3)
