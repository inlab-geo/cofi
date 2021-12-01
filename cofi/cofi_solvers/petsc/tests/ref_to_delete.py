import numpy
import matplotlib.pyplot as plt
from petsc4py import PETSc

"""PETSc does have a Levenberg-Marquardt optimizer.
it is a special case of its Bounded Regularized Gauss-Newton
"""


x=numpy.array([1,0.1,2,0.2,3,0.3])
t=numpy.linspace(0,10)
y=predict(x,t)
x0=numpy.array([2,0.2,3,0.3,4,0.1])
y0=predict(x0,t)
plt.plot(t,y)
plt.plot(t,y0)

class AppCtx(object):

    def __init__(self,t,y,x0):
        self.y  = y
        self.t = t 
        self.x0 = x0
        self.nm = len(x0)

    def evaluateJacobian(self, tao, x, J, Jp):
        jac = numpy.zeros([numpy.shape(self.t)[0],numpy.shape(x)[0]])
        for i in range(int(numpy.shape(x)[0]/2)):
            for j in range(len(self.t)):
                jac[j,i*2]=numpy.exp(-x[i*2+1]*t[j])
                jac[j,i*2+1]=-x[i*2]*t[j]*numpy.exp(-x[i*2+1]*t[j])
        J.setValues(range(0,len(self.t)), range(0,len(self.x0)), jac)   ## obj.jacobian(x)
        J.assemble()
        Jp.setValues(range(0,len(self.t)), range(0,len(self.x0)),  jac)   ## obj.jacobian(x) 
        Jp.assemble()

    def evaluateFunction(self, tao, x, f):
        yhat = numpy.zeros_like(self.t)
        for i in range(int(numpy.shape(x)[0]/2)):
            yhat += x[i*2]*numpy.exp(-x[i*2+1]*self.t) 
        f.setArray(yhat-self.y)            ## obj.residuals(x)
        f.assemble()

%%time

# access PETSc options database
OptDB = PETSc.Options()
OptDB.insertString("-tao_brgn_regularization_type lm") 

# create user application context and set the data and inital guess
user = AppCtx(t,y,x0)

# create solution vector, residual vector etc as PETSc vectors and matrices

x = PETSc.Vec().create(PETSc.COMM_SELF)
x.setSizes(user.nm)
x.setFromOptions()
x.setValues(range(0,len(x0)), x0) 

f = PETSc.Vec().create(PETSc.COMM_SELF)
f.setSizes(len(y))
f.setFromOptions()
f.set(0)

J = PETSc.Mat().createDense([len(t),len(x0)])
J.setFromOptions()
J.setUp()

Jp = PETSc.Mat().createDense([len(t),len(x0)])
Jp.setFromOptions()
Jp.setUp()


# create TAO Solver
tao = PETSc.TAO().create(PETSc.COMM_SELF)
tao.setType(PETSc.TAO.Type.BRGN)
#tao.setJacobian(user.evaluateJacobian, J)
tao.setResidual(user.evaluateFunction, f)
tao.setJacobianResidual(user.evaluateJacobian, J, Jp)


tao.setFromOptions()
x.setValues(range(0,len(x0)), x0) 
tao.solve(x)
x.view()



