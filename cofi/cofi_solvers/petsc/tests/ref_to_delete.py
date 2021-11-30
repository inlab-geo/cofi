"""
TAO solvers that work with an objective function and a gradient

"""


from petsc4py import PETSc
import numpy


class AppCtx(object):
    """Application context struct in C, similar to our Objective class
    """

    def __init__(self,t,y,x0):
        self.y  = y
        self.t = t 
        self.x0 = x0
        self.nm = len(x0)

        
    def formObjective(self, tao, x):
        #print ('formObjective')
        yhat = numpy.zeros_like(self.t)
        for i in range(int(numpy.shape(x)[0]/2)):
            yhat += x[i*2]*numpy.exp(-x[i*2+1]*self.t)
        return numpy.matmul((yhat-self.y),numpy.transpose(yhat-self.y))


    def formGradient(self, tao, x, G):
        #print ('formGradient')
        yhat = numpy.zeros_like(self.t)
        for i in range(int(numpy.shape(x)[0]/2)):
            yhat += x[i*2]*numpy.exp(-x[i*2+1]*self.t)
        jac = numpy.zeros([numpy.shape(self.t)[0],numpy.shape(x)[0]])
        for i in range(int(numpy.shape(x)[0]/2)):
            for j in range(len(self.t)):
                jac[j,i*2]=numpy.exp(-x[i+1]*self.t[j])
                jac[j,i*2+1]=-x[i*2]*self.t[j]*numpy.exp(-x[i+1]*self.t[j])
        G.setArray(numpy.matmul(numpy.transpose(jac),yhat-self.y))
        G.assemble()

      
    def formObjGrad(self, tao, x, G):
        #print ('formObjGrad')
        yhat = numpy.zeros_like(self.t)
        for i in range(int(numpy.shape(x)[0]/2)):
            yhat += x[i*2]*numpy.exp(-x[i*2+1]*self.t)
        jac = numpy.zeros([numpy.shape(self.t)[0],numpy.shape(x)[0]])
        for i in range(int(numpy.shape(x)[0]/2)):
            for j in range(len(self.t)):
                jac[j,i*2]=numpy.exp(-x[i+1]*self.t[j])
                jac[j,i*2+1]=-x[i*2]*self.t[j]*numpy.exp(-x[i+1]*self.t[j])
        G.setArray(numpy.matmul(numpy.transpose(jac),yhat-self.y))
        G.assemble()
        return numpy.matmul((yhat-self.y),numpy.transpose(yhat-self.y))

    def formHessian(self, tao, x, H, HP):
        #print ('formHessian')
        # Using the standard approximation (J^T J)
        jac = numpy.zeros([numpy.shape(self.t)[0],numpy.shape(x)[0]])
        for i in range(int(numpy.shape(x)[0]/2)):
            for j in range(len(self.t)):
                jac[j,i*2]=numpy.exp(-x[i+1]*self.t[j])
                jac[j,i*2+1]=-x[i*2]*self.t[j]*numpy.exp(-x[i+1]*self.t[j])
        
        Hessian=(numpy.matmul(numpy.transpose(jac),jac))
        H.setValues(range(0,self.nm), range(0,self.nm), Hessian) 
        H.assemble()


# access PETSc options database
OptDB = PETSc.Options()

# create user application context and set the data and inital guess
user = AppCtx(t,y,x0)

# create solution vector
x = PETSc.Vec().create(PETSc.COMM_SELF)
x.setSizes(user.nm)
x.setFromOptions()
x.setValues(range(0,len(x0)), x0) 


# create Hessian matrix
H = PETSc.Mat().create(PETSc.COMM_SELF)
H.setSizes([user.nm, user.nm])
H.setFromOptions()
H.setOption(PETSc.Mat.Option.SYMMETRIC, True)
H.setUp()


# use some of the methods available in PETSc 
# https://petsc.org/main/docs/manualpages/Tao/TaoSetType.html
methods=["nm","lmvm","nls","ntr","cg","blmvm","tron"]

for method in methods:
    # create TAO Solver
    tao = PETSc.TAO().create(PETSc.COMM_SELF)
    tao.setType(method)
    tao.setFromOptions()
    # solve the problem
    tao.setObjectiveGradient(user.formObjGrad)
    tao.setObjective(user.formObjective)
    tao.setGradient(user.formGradient)
    tao.setHessian(user.formHessian, H)
    x.setValues(range(0,len(x0)), x0) 
    tao.setInitial(x)
    tao.solve(x)
    print(method)
    x.view()
    tao.destroy()