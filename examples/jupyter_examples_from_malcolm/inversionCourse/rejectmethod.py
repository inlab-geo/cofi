# 1. rejectmethod
##################
import numpy as np
def rejectmethod(p,n):
    # Generate deviates from arbitrary 1-D PDF using the rejection method
    pmax=np.max(p[:,1])
    xmax=np.max(p[:,0])
    xmin=np.min(p[:,0])
    dev= np.ones(n)*xmin + (xmax-xmin)*np.random.rand(n)
    
    for i in range(len(dev)):
        J=[]
        K=[]
        
        for j in range(len(p)):
            if p[j,0]<dev[i]:
                J.append(j)
            if p[j,0]>dev[i]:
                K.append(j)
        jj=np.max(J)
        kk=np.min(K)
        pv=p[jj,1]+(dev[i]-p[jj,0])/(p[kk,0]-p[jj,0])*(p[kk,1]-p[jj,1])
        r=pmax*np.random.rand()
        if r > pv:
            dev[i]=-1
    rd=[]
    for i in range(len(dev)):
        if dev[i]>0.0:
            rd.append(dev[i])
    rd=np.array(rd)
    return rd