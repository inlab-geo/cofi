###################
import numpy as np
def eqlocate_orig(x0,y0,z0,ts,la,lo,el,vp,tol,solvedep=False):
    la2km=111.19
    lo2km=75.82
    
    i=5-1
    t0=ts[i]-np.sqrt(((x0*lo2km-lo[i]*lo2km)**2)+((y0*la2km-la[i]*la2km)**2)+(el[i]-z0)**2)/vp  # initial guess origin time
    
    ni=0
    sols=[[t0,x0,y0,z0]]
    ndata = len(ts) # Number of data
    
    while 1:
        ni=ni+1
        D0=[]
        for i in range(ndata):
            D0.append(np.sqrt((lo[i]*lo2km-x0*lo2km)**2+(la[i]*la2km-y0*la2km)**2+(el[i]-z0)**2))
        G=[]
        d=[]
        for i in range(ndata):
            if(solvedep):
                G.append([1,((x0-lo[i])*lo2km)/(D0[i]*vp),((y0-la[i])*la2km)/(D0[i]*vp),((z0-el[i]))/(D0[i]*vp)])
            else:
                G.append([1,((x0-lo[i])*lo2km)/(D0[i]*vp),((y0-la[i])*la2km)/(D0[i]*vp)])
            d.append(ts[i]-(D0[i]/vp+t0))
        G=np.array(G)
        d=np.array(d)
        m=np.linalg.lstsq(G,d)[0]
        t0=t0+m[0]
        x0=x0+m[1]/lo2km
        y0=y0+m[2]/la2km
        if(solvedep): 
            z0=z0+m[3]
            dtol = np.sqrt(m[1]**2+m[2]**2+m[3]**2) # distance moved by hypocentre
        else:
            dtol = np.sqrt(m[1]**2+m[2]**2)
        sols.append([t0,x0,y0,z0])        
        if m[0]<tol[0] and dtol<tol[1]:
            break
    sols=np.array(sols)
    return sols, d

def eqlocate(x0,y0,z0,ts,la,lo,el,vpin,tol,solvedep=False,nimax=100,verbose=False,kms2deg=[111.19,75.82]):
    la2km=kms2deg[0]
    lo2km=kms2deg[1]
    
    i=np.argmin(ts)
    #i = 4
    t0=ts[i]-np.sqrt(((x0*lo2km-lo[i]*lo2km)**2)+((y0*la2km-la[i]*la2km)**2)+(el[i]-z0)**2)/vpin[i]  # initial guess origin time
    
    ni=0
    sols=[[t0,x0,y0,z0]]
    ndata = len(ts) # Number of data
    
    while 1:
        ni=ni+1
        D0=np.zeros(ndata)
        for i in range(ndata):
            D0[i] = np.sqrt(((lo[i]-x0)*lo2km)**2+((la[i]-y0)*la2km)**2+(el[i]-z0)**2)
        G=[]
        res=[]
        for i in range(ndata):
            vp = vpin[i]
            if(solvedep):
                G.append([1,((x0-lo[i])*lo2km)/(D0[i]*vp),((y0-la[i])*la2km)/(D0[i]*vp),(z0-el[i])/(D0[i]*vp)])
            else:
                G.append([1,((x0-lo[i])*lo2km)/(D0[i]*vp),((y0-la[i])*la2km)/(D0[i]*vp)])
            res.append(ts[i]-(D0[i]/vp+t0))
        G=np.array(G)
        res=np.array(res)
        #print(' ni ',ni)
        #print('G :\n',G[ni-1])
        #print('d :\n',d[ni-1])
        m=np.linalg.lstsq(G,res)[0]
        t0=t0+m[0]
        x0=x0+m[1]/lo2km # update longitude solution and convert to degrees
        y0=y0+m[2]/la2km # update latitude solution and convert to degrees
        if(solvedep): 
            z0=z0+m[3]
            dtol = np.sqrt((m[1]**2+m[2]**2+m[3]**2)) # distance moved by hypocentre
        else:
            dtol = np.sqrt(m[1]**2+m[2]**2)
        chisq = np.dot(res.T,res)
        if(verbose): print('Iteration :',ni,'Chi-sq:',chisq,' Change in origin time',m[0],' change in spatial distance:',dtol)
        sols.append([t0,x0,y0,z0])        
        if m[0]<tol[0] and dtol<tol[1]:
            break
        if(ni==nimax):
            print(' Maximum number of iterations reached in eqlocate. No convergence')
            break
    sols=np.array(sols)
    return sols, res