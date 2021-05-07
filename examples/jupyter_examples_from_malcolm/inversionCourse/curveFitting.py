import numpy as np

# Andrew Valentine (andrew.valentine@anu.edu.au)
# Malcolm Sambridge (malcolm.sambridge@anu.edu.au)
#
# Research School of Earth Sciences
# The Australian National University
#
# May 2018

def curveFittingFwd(model, xpts, basis='polynomial',domainLength = 1.):
    singlePoint = False
    if domainLength<=0: raise ValueError("Argument 'domainLength' must be positive")
    # Convert list inputs to arrays
    if type(model) is type([]): model = np.array(model)
    if type(xpts) is type([]): xpts = np.array(xpts)
    # Check model parameters
    try:
        nModelParameters = model.shape[0]
    except AttributeError:
        raise ValueError("Argument 'model' should be a 1-D array")
    if len(model.shape)>1: raise ValueError("Argument 'model' should be a 1-D array")
    # Check x-values
    try:
        npts = xpts.shape[0]
        if len(xpts.shape)!=1: raise ValueError("Argument 'xpts' should be a 1-D array")
    except AttributeError:
        singlePoint = True
        npts = 1
    if basis == 'polynomial':
        # y = m[0] + m[1]*x + m[2]*x**2 +...
        y = model[0]*np.ones([npts])
        for iord in range(1,nModelParameters):
            y += model[iord]*xpts**iord
    elif basis == 'fourier':
        if nModelParameters%2==0: raise ValueError("Fourier basis requires odd number of model parameters")
        if not np.all(0<= xpts) and np.all(xpts<= domainLength): raise ValueError("For Fourier basis, all sample points must be in interval (0,domainLength)")
        # y = m[0]/2 + m[1] sin (pi x/L) + m[2] cos (pi x/L) + m[3] sin(pi x/L) + ...
        y = np.ones([npts])*0.5*model[0]
        n = 1
        for i in range(1,nModelParameters,2):
            y += model[i]*np.sin(2*n*np.pi*xpts/domainLength) + model[i+1]*np.cos(2*n*np.pi*xpts/domainLength)
            n+=1
    elif basis == 'discrete':
        if not np.all(0<= xpts) and np.all(xpts<= domainLength): raise ValueError("For discrete basis, all sample points must be in interval (0,domainLength)")
        bounds = np.linspace(0,domainLength,nModelParameters+1)
        y = np.zeros([npts])
        for ipt in range(0,npts):
            y[ipt] = model[max(0,np.searchsorted(bounds,xpts[ipt])-1)]
    if singlePoint:
        return y[0]
    else:
        return y

def curveFittingInv(xpts,ypts,nModelParameters, basis='polynomial',domainLength=1.,regularisation=None,priorModel=None,returnPosteriorCovariance=False):
    singlePoint=False
    if domainLength<0:raise ValueError("Argument 'domainLength' must be positive")
    if type(xpts) is type([]):xpts=np.array(xpts)
    if type(ypts) is type([]):ypts=np.array(ypts)
    try:
        npts = xpts.shape[0]
        if len(xpts.shape)!=1: raise ValueError("Argument 'xpts' should be a 1-D array")
    except AttributeError:
        singlePoint = True
        npts = 1
    try:
        if ypts.shape[0] != npts: raise ValueError("Argument 'ypts' should have same dimension as 'xpts'")
        if len(ypts.shape)!=1: raise ValueError("Argument 'ypts' should be a 1-D array")
    except AttributeError:
        if not singlePoint: raise ValueError("Argument 'ypts' should have same dimension as 'xpts'")

    if basis == 'polynomial':
        G = np.zeros([npts,nModelParameters])
        for iord in range(0,nModelParameters):
            G[:,iord] = xpts**iord
    elif basis == 'fourier':
        if nModelParameters%2==0: raise ValueError("Fourier basis requires an odd number of model parameters")
        G = np.zeros([npts,nModelParameters])
        G[:,0] = 0.5
        n = 1
        for i in range(1,nModelParameters,2):
            G[:,i] = np.sin(2*n*np.pi*xpts/domainLength)
            G[:,i+1] = np.cos(2*n*np.pi*xpts/domainLength)
            n+=1
    elif basis == 'discrete':
        G = np.zeros([npts, nModelParameters])
        bounds = np.linspace(0,domainLength,nModelParameters+1)
        y = np.zeros([npts])
        for ipt in range(0,npts):
            G[ipt,max(0,np.searchsorted(bounds,xpts[ipt])-1)] = 1.
    GTG = G.T.dot(G)
    if regularisation is not None:
        if regularisation<0: raise ValueError("Argument 'regularisation' should be positive or None")
        GTG+=regularisation*np.eye(GTG.shape[0])
        if priorModel is None:
            mp = np.zeros(nModelParameters)
        else:
            if type(priorModel) is type([]): priorModel = np.array(priorModel)
            try:
                if priorModel.shape[0]!=nModelParameters: raise ValueError ("Argument 'priorModel' should match requested number of model parameters")
                if len(priorModel.shape)!=1: raise ValueError ("Argument 'priorModel' should be a 1-D array")
            except AttributeError:
                if nModelParameters>1:raise ValueError("Argument 'priorModel' should match requested number of model parameters")
            mp = priorModel
    else:
        mp = np.zeros(nModelParameters)
    if returnPosteriorCovariance:
        return mp+np.linalg.inv(GTG).dot(G.T.dot(ypts-G.dot(mp))), np.linalg.inv(GTG)
    else:
        return mp+np.linalg.inv(GTG).dot(G.T.dot(ypts-G.dot(mp)))

def generateExampleDatasets():
    np.random.seed(42)
    # Example 1: Straight line
    model = np.array([0.5,2])
    xpts = np.random.uniform(0,1,size=20)
    xpts.sort()
    ypts = curveFittingFwd(model,xpts,'polynomial')+np.random.normal(0,0.1,size=20)
    fp = open('curve_fitting_1.dat','w')
    fp.write('# x       y        sigma\n')
    for i in range(0,20):
        fp.write("%.3f    %.3f    %.3f\n"%(xpts[i],ypts[i],0.1))
    fp.close()

    # Example 2: Cubic
    model = np.array([1.,-0.2,0.5,0.3])
    xpts = np.random.uniform(0,1,size=25)
    xpts.sort()
    ypts = curveFittingFwd(model,xpts,'polynomial')+np.random.normal(0,0.1,size=25)
    fp = open('curve_fitting_2.dat','w')
    fp.write('# x       y        sigma\n')
    for i in range(0,25):
        fp.write("%.3f    %.3f    %.3f\n"%(xpts[i],ypts[i],0.1))
    fp.close()


    # Example 3: Sinusoid
    model = np.array([1,0,0.3,-0.2,0,0,0.7,0.,0.,0.3,0.,0.,0.,0.,-.2])
    xpts = np.random.uniform(0,1,size=50)
    xpts.sort()
    ypts = curveFittingFwd(model,xpts,'fourier')+np.random.normal(0,0.1,size=50)
    fp = open('curve_fitting_3.dat','w')
    fp.write('# x       y        sigma\n')
    for i in range(0,50):
        fp.write("%.3f    %.3f    %.3f\n"%(xpts[i],ypts[i],0.1))
    fp.close()

    # Example 4: Small dataset
    model = np.array([0.,2,-.3,-.6])
    xpts = np.random.uniform(0,1,size=5)
    xpts.sort()
    ypts = curveFittingFwd(model,xpts,'polynomial')+np.random.normal(0,0.05,size=5)
    fp = open('curve_fitting_4.dat','w')
    fp.write('# x       y        sigma\n')
    for i in range(0,5):
        fp.write("%.3f    %.3f    %.3f\n"%(xpts[i],ypts[i],0.05))
    fp.close()

    # Example 5: Incomplete dataset
    model = np.array([0.3,0.3,0.,-0.2,0.5,-.8,0.1,0.125])
    xpts = np.zeros([30])
    xpts[0:25] = np.random.uniform(0,0.3,size=25)
    xpts[25:] = np.random.uniform(0.9,1.,size=5)
    xpts.sort()
    ypts = curveFittingFwd(model,xpts,'polynomial')+np.random.normal(0,0.1,size=30)
    fp = open('curve_fitting_5.dat','w')
    fp.write('# x       y        sigma\n')
    for i in range(0,30):
        fp.write("%.3f    %.3f    %.3f\n"%(xpts[i],ypts[i],0.1))
    fp.close()


    return xpts,ypts
