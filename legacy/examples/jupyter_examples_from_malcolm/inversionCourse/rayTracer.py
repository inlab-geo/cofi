import numpy as np
import scipy.optimize as optim
import tqdm
import matplotlib.pyplot as plt
from PIL import Image

def norm(x):
    return np.sqrt(x.dot(x))
def normalise(x):
    return x/norm(x)
def pngToModel(pngfile,nx,ny,bg=1.,sc=1.):
    png = Image.open(pngfile)
    png.load()

    model = sc*(bg+np.asarray(png.convert('L').resize((nx,ny)).transpose(Image.ROTATE_270))/255.)
    return model

def displayModel(model,paths=None,extent=(0,1,0,1),clim=None,cmap=None,figsize=(6,6),title=None,line=1.0):
    plt.figure(figsize=figsize)
    if cmap is None: cmap = plt.cm.RdBu

    plt.imshow(model.T,origin='lower',extent=extent,cmap=cmap)


    if paths is not None:
        for p in paths:
            plt.plot(p[:,0],p[:,1],'k',lw=line)
    if clim is not None: plt.clim(clim)
    
    if title is not None: plt.title(title)
    
    plt.colorbar()

    plt.show()

def generateSurfacePoints(nPerSide,extent=(0,1,0,1),surface=[True,True,True,True],addCorners=True):
    out = []
    if surface[0]:
        out+=[[extent[0],x] for x in np.linspace(extent[2],extent[3],nPerSide+2)[1:nPerSide+1]]
    if surface[1]:
        out+=[[extent[1],x] for x in np.linspace(extent[2],extent[3],nPerSide+2)[1:nPerSide+1]]
    if surface[2]:
        out+=[[x,extent[2]] for x in np.linspace(extent[0],extent[1],nPerSide+2)[1:nPerSide+1]]
    if surface[3]:
        out+=[[x,extent[3]] for x in np.linspace(extent[0],extent[1],nPerSide+2)[1:nPerSide+1]]
    if addCorners:
        if surface[0] or surface[2]:
            out+=[[extent[0],extent[2]]]
        if surface[0] or surface[3]:
            out+=[[extent[0],extent[3]]]
        if surface[1] or surface[2]:
            out+=[[extent[1],extent[2]]]
        if surface[1] or surface[3]:
            out+=[[extent[1],extent[3]]]
    return np.array(out)
class gridModel(object):
    def __init__(self,velocities,extent=(0,1,0,1)):
        self.nx,self.ny = velocities.shape
        self.velocities=velocities
        self.xmin,self.xmax,self.ymin,self.ymax = extent
        self.xx = np.linspace(self.xmin,self.xmax,self.nx+1)
        self.yy = np.linspace(self.ymin,self.ymax,self.ny+1)
    def getVelocity(self):
        return self.velocities.copy()
    def getSlowness(self):
        return 1./self.velocities # No copy needed as operation must return copy
    def setVelocity(self,v):
        assert self.velocities.shape == v.shape
        self.velocities = v.copy()
    def setSlowness(self,s):
        assert self.velocities.shape == s.shape
        self.velocities = 1./s
    def convertToGridCoords(self,x):
        return np.array([float(self.nx)*(x[0]-self.xmin)/float(self.xmax-self.xmin), float(self.ny)*(x[1]-self.ymin)/float(self.ymax-self.ymin)])
    def convertFromGridCoords(self,p):
        return np.array([self.xmin+p[0]*float(self.xmax-self.xmin)/float(self.nx), self.ymin+p[1]*float(self.ymax-self.ymin)/float(self.ny)])
    def lengthFromGridCoords(self,v):
        return norm(self.convertFromGridCoords(v))
    def cellTrace(self,ix,iy,traceFromPoint,traceDirection,borders=(True,True,True,True)):
        chkLeft,chkRight,chkBottom,chkTop = borders

        if chkLeft:
            if traceDirection[0] ==0:
                lamLeft = np.Inf
            else:
                lamLeft = (ix - traceFromPoint[0])/traceDirection[0]
        else:
            lamLeft = None
        if chkRight:
            if traceDirection[0] == 0:
                lamRight = np.Inf
            else:
                lamRight = (ix+1 - traceFromPoint[0])/traceDirection[0]
        else:
            lamRight = None
        if chkBottom:
            if traceDirection[1] == 0:
                lamBottom = np.Inf
            else:
                lamBottom = (iy - traceFromPoint[1])/traceDirection[1]
        else:
            lamBottom = None
        if chkTop:
            if traceDirection[1] == 0:
                lamTop = np.Inf
            else:
                lamTop = (iy+1 - traceFromPoint[1])/traceDirection[1]
        else:
            lamTop = None
        return (lamLeft,lamRight,lamBottom,lamTop)
    def gridTrace(self,traceFromPoint,traceDirection,returnPath=False,returnA=False):
        #print "Tracing:", traceFromPoint,traceDirection
        thresh=1.e-12
        path = []
        traceFromPoint = self.convertToGridCoords(traceFromPoint)
        traceDirection = normalise(self.convertToGridCoords(traceDirection))
        #print traceFromPoint, traceDirection
        xUpdates = np.array([])
        ix,iy,chks = self.locate(traceFromPoint[0],traceFromPoint[1],traceDirection[0],traceDirection[1])

        tt=0.
        if returnA: A = np.zeros_like(self.velocities)
        loopCount=0
        while True:
            loopCount+=1
            if loopCount>10*self.nx*self.ny:
                print (path)
                raise RuntimeError("gridTrace appears to have entered an infinite loop...")
            if returnPath:path+=[self.convertFromGridCoords(traceFromPoint)]
            #print traceFromPoint,ix,iy
            lams = self.cellTrace(ix,iy,traceFromPoint,traceDirection,chks)
            iNearest = 0
            iOtherNearest = 0
            hitCorner=False
            #print lams
            while lams[iNearest] is None or lams[iNearest]<=0:
                iNearest+=1
                if iNearest == 4: raise ValueError("Logic error")
            for i in range(iNearest+1,4):
                if lams[i] is None: continue
                if 0<lams[i]<lams[iNearest]: iNearest = i
            for i in range(0,4):
                if i == iNearest: continue
                if lams[i] is None:continue
                if abs(lams[i]-lams[iNearest])<thresh:
                    if hitCorner: raise ValueError("Logic Error")
                    hitCorner=True
                    iOtherNearest = i
            #print lams
            if hitCorner:
                #print "Corner"
                xNearest = min(iNearest,iOtherNearest)
                yNearest = max(iNearest,iOtherNearest)-2
                # Line continues straight
                tt+=self.lengthFromGridCoords(np.array([ix+xNearest,iy+yNearest])-traceFromPoint)/self.velocities[ix,iy]
                if returnA: A[ix,iy]+=self.lengthFromGridCoords(np.array([ix+xNearest,iy+yNearest])-traceFromPoint)
                traceFromPoint = np.array([ix+xNearest,iy+yNearest])
                if xNearest==0:
                    ix+= -1
                    chks[0] = True
                    chks[1] = False
                else:
                    ix+=1
                    chks[0] = False
                    chks[1] = True
                if yNearest == 0:
                    iy +=-1
                    chks[2] = True
                    chks[3] = False
                else:
                    iy += 1
                    chks[2] = False
                    chks[3] = True
                if ix<0 or ix==self.nx or iy<0 or iy==self.ny: break
            else:
                if iNearest==0:
                    # Exit cell left
                    tt+=self.lengthFromGridCoords(np.array([ix,traceFromPoint[1]+lams[iNearest]*traceDirection[1]])-traceFromPoint)/self.velocities[ix,iy]
                    if returnA: A[ix,iy]+=self.lengthFromGridCoords(np.array([ix,traceFromPoint[1]+lams[iNearest]*traceDirection[1]])-traceFromPoint)
                    traceFromPoint = np.array([ix,traceFromPoint[1]+lams[iNearest]*traceDirection[1]])
                    if ix-1<0:
                        break
                    else:
                        sth1 = self.velocities[ix-1,iy]*traceDirection[1]/self.velocities[ix,iy]
                        if abs(sth1)==1:
                            raise NotImplementedError("Ray exits along gridline")
                        elif abs(sth1)>1:
                            #print "***",sth1,ix,iy,self.velocities[ix-1,iy],traceDirection[1],self.velocities[ix,iy]
                            traceDirection[0] = -traceDirection[0]
                            chks = [False,True,True,True]
                        else:
                            traceDirection[1] = sth1
                            traceDirection[0] = -np.sqrt(1-sth1**2)
                            ix += -1
                            chks = [True,False,True,True]
                elif iNearest==1:
                    # Exit cell right
                    tt+=self.lengthFromGridCoords(np.array([ix+1,traceFromPoint[1]+lams[iNearest]*traceDirection[1]])-traceFromPoint)/self.velocities[ix,iy]
                    if returnA: A[ix,iy]+=self.lengthFromGridCoords(np.array([ix+1,traceFromPoint[1]+lams[iNearest]*traceDirection[1]])-traceFromPoint)
                    traceFromPoint = np.array([ix+1,traceFromPoint[1]+lams[iNearest]*traceDirection[1]])
                    if ix+1==self.nx:
                        break
                    else:
                        sth1 = self.velocities[ix+1,iy]*traceDirection[1]/self.velocities[ix,iy]
                        if abs(sth1)==1:
                            raise NotImplementedError("Ray exits along gridline")
                        elif abs(sth1)>1:
                            #print "***",sth1,ix,iy,self.velocities[ix+1,iy],traceDirection[1],self.velocities[ix,iy]
                            traceDirection[0] = -traceDirection[0]
                            chks = [True,False,True,True]
                        else:
                            traceDirection[1] = sth1
                            traceDirection[0] = np.sqrt(1-sth1**2)
                            ix += 1
                            chks = [False,True,True,True]
                elif iNearest==2:
                    # Exit cell bottom
                    tt+=self.lengthFromGridCoords(np.array([traceFromPoint[0]+lams[iNearest]*traceDirection[0],iy])-traceFromPoint)/self.velocities[ix,iy]
                    if returnA: A[ix,iy]+=self.lengthFromGridCoords(np.array([traceFromPoint[0]+lams[iNearest]*traceDirection[0],iy])-traceFromPoint)
                    traceFromPoint = np.array([traceFromPoint[0]+lams[iNearest]*traceDirection[0],iy])
                    if iy-1<0:
                        break
                    else:
                        sth1 = self.velocities[ix,iy-1]*traceDirection[0]/self.velocities[ix,iy]
                        if abs(sth1)==1:
                            raise NotImplementedError("Ray exits along gridline")
                        elif abs(sth1)>1:
                            #print "***",sth1,ix,iy,self.velocities[ix,iy-1],traceDirection[0],self.velocities[ix,iy]
                            traceDirection[1]= -traceDirection[1]
                            chks = [True,True,False,True]
                        else:
                            traceDirection[0] = sth1
                            traceDirection[1] = -np.sqrt(1-sth1**2)
                            iy += -1
                            chks = [True,True,True,False]
                elif iNearest==3:
                    # Exit cell top
                    tt+=self.lengthFromGridCoords(np.array([traceFromPoint[0]+lams[iNearest]*traceDirection[0],iy+1])-traceFromPoint)/self.velocities[ix,iy]
                    if returnA: A[ix,iy]+=self.lengthFromGridCoords(np.array([traceFromPoint[0]+lams[iNearest]*traceDirection[0],iy+1])-traceFromPoint)
                    traceFromPoint = np.array([traceFromPoint[0]+lams[iNearest]*traceDirection[0],iy+1])
                    if iy+1==self.ny:
                        break
                    else:
                        sth1 = self.velocities[ix,iy+1]*traceDirection[0]/self.velocities[ix,iy]
                        if abs(sth1)==1:
                            raise NotImplementedError("Ray exits along gridline")
                        elif abs(sth1)>1:
                            #print "***",sth1,ix,iy,self.velocities[ix,iy+1],traceDirection[0],self.velocities[ix,iy]
                            traceDirection[1] = -traceDirection[1]
                            chks = [True,True,True,False]
                        else:
                            traceDirection[0] = sth1
                            traceDirection[1] = np.sqrt(1-sth1**2)
                            iy += 1
                            chks = [True,True,False,True]
        #print traceFromPoint
        if returnPath:
            path+=[self.convertFromGridCoords(traceFromPoint)]
            if returnA:
                return self.convertFromGridCoords(traceFromPoint),tt,np.array(path),A
            else:
                return self.convertFromGridCoords(traceFromPoint),tt,np.array(path)
        else:
            if returnA:
                return self.convertFromGridCoords(traceFromPoint),tt,A
            else:
                return self.convertFromGridCoords(traceFromPoint),tt
    def locate(self,x,y,xdir=1,ydir=1):
        cchks = [True,True,True,True]
        if int(x)==x:
            if xdir>=0:
                ix = int(x)
                cchks[0]=False
            else:
                ix = int(x)-1
                cchks[1]=False
            if ix<0:ix = 0
            if ix==self.nx: ix = self.nx-1
        else:
            ix = int(x)
        if int(y)==y:
            if ydir>=0:
                iy = int(y)
                cchks[2] = False
            else:
                iy = int(y)-1
                cchks[3] = False
            if iy<0:iy = 0
            if iy==self.ny: iy = self.ny-1
        else:
            iy = int(y)
        return ix,iy,cchks
    # def findPath(self,src,rec,nShoot=180):
    #     thShoot = np.linspace(0,2*np.pi,nShoot+1)[0:nShoot]
    #     dists = np.zeros([nShoot])
    #     for iShot in range(nShoot):
    #         p,tt = self.gridTrace(src,np.array([np.cos(thShoot[iShot]),np.sin(thShoot[iShot])]))
    #         dists[iShot] = bdist(p,rec,(self.xmin,self.xmax,self.ymin,self.ymax))
    #         print iShot,thShoot[iShot],p,dists[iShot]
    #     #print dists
    #     brackets=[]
    #     for iShot in range(-1,nShoot-1):
    #         if dists[iShot]*dists[iShot+1]<0 and abs(dists[iShot]-dists[iShot+1])<(self.xmax+self.ymax-self.xmin-self.ymin):
    #             brackets+=[[thShoot[iShot],thShoot[iShot+1]]]
    #     ttimes = np.zeros([len(brackets)])
    #
    #     paths=[]
    #     for ib,b in enumerate(brackets):
    #         print b
    #         th,rtfind = optim.bisect(lambda x: bdist(self.gridTrace(src,np.array([np.cos(x),np.sin(x)]))[0],rec,extent=(self.xmin,self.xmax,self.ymin,self.ymax)),b[0],b[1],full_output=True)
    #         print "Getting paths:"
    #         p,tt,path = self.gridTrace(src,np.array([np.cos(th),np.sin(th)]),returnPath=True)
    #         print rtfind,rtfind.converged,bdist(path[-1],rec,(self.xmin,self.xmax,self.ymin,self.ymax))
    #         print "Done"
    #         ttimes[ib] =tt
    #         paths+=[path]
    #     imin = np.argmin(ttimes)
    #     return ttimes,paths
    def shootInitial(self,src,thlo=0.,thhi=2.*np.pi,nShoot=360):
        self.src = src
        self.thShoot = np.linspace(thlo,thhi,nShoot)
        self.exits = np.zeros([nShoot,2])
        self.tts = np.zeros([nShoot])
        for iShot in range(nShoot):
            self.exits[iShot,:],self.tts[iShot] = self.gridTrace(src,np.array([np.cos(self.thShoot[iShot]), \
                                                                                np.sin(self.thShoot[iShot])]))


    def shootSupplemental(self,theta,returnPathA=False):
        itheta=np.searchsorted(self.thShoot,theta,'left')
        n=self.thShoot.shape[0]
        if returnPathA:
            p,tt,path,A = self.gridTrace(self.src,np.array([np.cos(theta),np.sin(theta)]),returnPath=True,returnA=True)
        else:
            p,tt = self.gridTrace(self.src,np.array([np.cos(theta),np.sin(theta)]))
        self.thShoot = np.insert(self.thShoot,itheta,theta)
        tmp = np.insert(self.exits.flatten(),2*itheta,p)
        self.exits = tmp.reshape(n+1,2)
        self.tts = np.insert(self.tts,itheta,tt)
        if returnPathA:
            return p,tt,path,A
        else:
            return p,tt
    def findPath(self,rec,shootThresh=1e-5,nBracket=10,acceptThresh=1e-3):
        i = 0
        disti = bdist(rec,self.exits[0,:],(self.xmin,self.xmax,self.ymin,self.ymax))
        tts=[]
        paths=[]
        AA = []
        while True:
            if i+1>=self.thShoot.shape[0]: break
            distip1 = bdist(rec,self.exits[i+1,:],(self.xmin,self.xmax,self.ymin,self.ymax))
            if disti==0.:
                p,tt,path,A = self.gridTrace(self.src,np.array([np.cos(self.thShoot[i]),np.sin(self.thShoot[i])]),returnPath=True,returnA=True)
                tts+=[tt]
                paths +=[path]
                AA+=[A]
                i+=1
                disti=distip1
                continue
            elif -0.25<disti*distip1<0:
                if abs(self.thShoot[i+1]-self.thShoot[i])>shootThresh:
                    for th in np.linspace(self.thShoot[i],self.thShoot[i+1],nBracket+2)[1:nBracket+1]:
                        self.shootSupplemental(th)
                    continue
                else:
                    #p,tt,path = self.shootSupplemental(0.5*(self.thShoot[i]+self.thShoot[i+1]),returnPath=True)
                    #print i,i+1,self.thShoot[i],self.thShoot[i+1],self.exits[i,:],self.exits[i+1],bdist(rec,self.exits[i,:],(0,1,0,1)),bdist(rec,self.exits[i+1,:],(0,1,0,1)),self.gridTrace(self.src,np.array([np.cos(self.thShoot[i]),np.sin(self.thShoot[i])]))[0]
                    th = optim.brentq(lambda th: bdist(rec,self.gridTrace(self.src,np.array([np.cos(th),np.sin(th)]))[0],(self.xmin,self.xmax,self.ymin,self.ymax)),self.thShoot[i],self.thShoot[i+1])
                    p,tt,path,A = self.shootSupplemental(th,returnPathA=True)
                    dist = bdist(rec,p,(self.xmin,self.xmax,self.ymin,self.ymax))
                    if norm(rec-p) < acceptThresh: #abs(dist)<acceptThresh:
                        tts+=[tt]
                        paths+=[path]
                        AA+=[A]
                    #else:
                    #        print "Fail:",dist,disti,distip1,disti*distip1, norm(rec-p)
                    i+=2
                    disti = distip1
                    continue
            disti=distip1
            i+=1
        return tts,paths,AA
    def tracer(self,srcs,recs):
        tts = np.zeros(srcs.shape[0]*recs.shape[0])
        AA = np.zeros([srcs.shape[0]*recs.shape[0],self.nx*self.ny])
        i = 0
        t=tqdm.tqdm(total = srcs.shape[0]*recs.shape[0])
        for src in srcs:
            self.shootInitial(src)
            for rec in recs:
                try:
                    pathtimes,paths,A = self.findPath(rec)
                except NotImplementedError:
                    # Skip this path and move on
                    i+=1
                    t.update(1)
                    continue
                if len(pathtimes)==0:
                    i+=1
                    t.update(1)
                    continue
                elif len(pathtimes)==1:
                    tts[i] = pathtimes[0]
                    AA[i,:] = A[0].flatten()
                else:
                    imin = np.argmin(np.array(pathtimes))
                    tts[i] = pathtimes[imin]
                    AA[i,:] = A[imin].flatten()
                i+=1
                t.update(1)
        t.close()
        return tts,AA

def bdist_orig(p,extent,tol=1e-12):
    if abs(p[0]-extent[0])<tol:
        d =2*(extent[1]+extent[3])-p[1]
    elif abs(p[0]-extent[1])<tol:
        d = extent[1]+p[1]
    elif abs(p[1]-extent[2])<tol:
        d = p[0]
    elif abs(p[1]-extent[3])<tol:
        d = 2*extent[1]+extent[3] - p[0]
    else:
        print (p,extent)
        print (abs(p[0]-extent[0]),abs(p[0]-extent[1]),abs(p[1]-extent[2]),abs(p[1]-extent[3]))
        raise ValueError('p must be on boundary')
    return d
def bdist(p,q,extent):
    xmin,xmax,ymin,ymax=extent
    xc = 0.5*(xmax-xmin)
    yc = 0.5*(ymax-ymin)
    thp = np.arctan2(p[1]-yc,p[0]-xc)
    thq = np.arctan2(q[1]-yc,q[0]-xc)
    #print p,q,thq-thp
    return thq-thp
    #return bdist_orig(q,extent)-bdist_orig(p,extent)
