#-----------------------------------------------------------------------
# This script is an interface between Receiver function routines and python
# 
# Most parameters relating to the Receiver Function are set within the Fortran routine.
#
# The fortran interface was generated with f2py using the call
# > f2py -m RF -c RF.F90 RFsubs/*.f*
#
# Fortran routines for Receiver function calculation by T. Shibutani
#
# M. Sambridge, 
# RSES, Oct., 2017.
#
#-----------------------------------------------------------------------
#import sys
try:
    from .. import _rfc as rfm # if file is only to be used as library
except:
    import _rfc as rfm # if file is to be executed
import numpy as np
#matplotlib.use('Qt4Agg')
#import matplotlib
from matplotlib import pyplot as plt
##################################################################################
def rfcalc(model,sn=0.0,mtype=0,fs=25.0,gauss_a=2.5,water_c=0.0001,angle=35.0,time_shift=5.0,ndatar=626,v60=8.043,seed=1): # Calculate Data covariance matrix for evaluation of waveform fit
    #print ('sn ',sn)
    if(sn==0.0):
        a,b = rfm.rfcalc_nonoise(model,mtype,fs,gauss_a,water_c,angle,time_shift,ndatar,v60)
    else:
        a,b = rfm.rfcalc_noise(model,mtype,sn,fs,gauss_a,water_c,angle,time_shift,ndatar,v60,seed)
    return a,b
##################################################################################
def InvDataCov(width,Ar,ndata): # Calculate Data covariance matrix for evaluation of waveform fit
    sigsq = width**2
    Arsq = Ar*Ar # waveform noise variance    
    Cd = np.zeros((ndata,ndata))
    for k1 in range(ndata):
        for k2 in range(ndata):
            Cd[k1][k2] = np.exp(-0.5*(k1-k2)**2/sigsq)
    U, s, V = np.linalg.svd(Cd)
    pmax = [ n for n,a in enumerate(s) if a<0.000001 ][0]
    Sinv = np.diag(1.0/s[:pmax])
    Cdinv = np.dot(V.T[:,:pmax], np.dot(Sinv, U.T[:pmax,:]))
    return Cdinv/Arsq

##################################################################################
def InvDataCovSub(width,Ar,ndata,ind): # Calculate Data sub-covariance matrix for evaluation of waveform fit
    sigsq = width**2
    Arsq = Ar*Ar # waveform noise variance    
    Cd = np.zeros((ndata,ndata))
    for k1 in range(ndata):
        for k2 in range(ndata):
            Cd[k1][k2] = np.exp(-0.5*(k1-k2)**2/sigsq)
    Cdused = Cd[ind,:][:,ind]
    U, s, V = np.linalg.svd(Cdused)
    pmax = [ n for n,a in enumerate(s) if a>0.000001 ][-1]+1
    Sinv = np.diag(1.0/s[:pmax])
    Cdinv = np.dot(V.T[:,:pmax], np.dot(Sinv, U.T[:pmax,:]))
    return Cdinv/Arsq        
##################################################################################
def plot_misfit_profile(x,misfit,xtrue,iparam): # Plot calculated and observed RF waveform 
    fig, ax = plt.subplots()
    
    if(iparam[1]==1): strx = "Layer "+repr(iparam[0]+1)+" velocity (km/s)" # plot axis labels
    if(iparam[1]==0): strx = "Layer "+repr(iparam[0]+1)+" node depth (km)"

    mismin = np.min(misfit)
    mismax = np.max(misfit)
    xt = [xtrue,xtrue]
    yt = [mismin,mismin+0.1*(mismax-mismin)]        
    ax.plot(x, misfit, 'k-')
    ax.plot(xt,yt,'r-',label='True')
    if(iparam[1] == 0):
        ax.set_xlabel('Node Depth (km)') 
    else:
        ax.set_xlabel('Velocity (km)/s')
    ax.set_xlabel(strx)
    #ax.set_ylabel('-Log L(m)')
    #ax.set_title("-Log-Likelihood through reference model")
    ax.set_ylabel('Misfit')
    ax.set_title("Waveform misfit profile")
    ax.grid(True)
    xt2 = [x[np.argmin(misfit)],x[np.argmin(misfit)]] 
    ax.plot(xt2,yt,'c-',label='Minimum') 
    plt.title("Misfit profile along single axis")
    plt.legend()
    plt.show()
##################################################################################
def l2mod(model,vmin=2.4,vmax=4.7,dmin=0.0,dmax=60.0): # Transform layer thickness representation to (depth vel) plot format
    px = np.zeros([2*len(model)])
    py = np.zeros([2*len(model)])
    py[1::2],py[2::2],px[0::2],px[1::2] = list(np.cumsum(model.T[0])),list(np.cumsum(model.T[0]))[:-1],model.T[1],model.T[1]
    py[-1] = dmax
    return px,py
##################################################################################
def v2mod(model,vmin=2.4,vmax=4.7,dmin=0.0,dmax=60.0): # Transform Voronoi nucleus representation to (depth vel) plot format
    a,b,c,d,e = rfm.voro2mod(model)
    px = np.zeros([2*len(d)])
    py = np.zeros([2*len(d)])
    a,b,c,d,e = rfm.voro2mod(model)
    py[1::2],py[2::2],px[0::2],px[1::2] = list(np.cumsum(a)),list(np.cumsum(a))[:-1],b[:],b[:]
    py[-1] = dmax
    return np.cumsum(a),b,c,d,e,px,py
##################################################################################
def d2mod(model,vmin=2.4,vmax=4.7,dmin=0.0,dmax=60.0): # Transform depth representation to (depth vel) plot format
    px = np.zeros([2*len(model)])
    py = np.zeros([2*len(model)])
    py[1::2],py[2::2],px[0::2],px[1::2] = list(model.T[0]),list(model.T[0])[:-1],model.T[1],model.T[1]
    py[-1] = dmax
    return px,py
##################################################################################
def plot_RFs(time1,RFo,time2,RFp,string="Observed and predicted receiver functions"):
    fig, ax = plt.subplots()
    plt.title(string)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.plot(time1, RFo, 'k-', label='Observed')
    plt.plot(time2, RFp, 'r-', label='Predicted')
    plt.legend()
    plt.show()
    plt.savefig('RF_plot.pdf',format='PDF')
##################################################################################
def plotRFm(velmod,time1,RFo,time2,RFp,mtype = 0,vmin=2.4,vmax=4.7,dmin=0.0,dmax=60.0,string="Observed and predicted receiver functions"): # plot velocity model and receiver function
    f, (a0, a1) = plt.subplots(1,2, figsize=(12,4), gridspec_kw = {'width_ratios':[1, 3]})

    a1.set_title(string)
    a1.set_xlabel("Time (s)")
    a1.set_ylabel("Amplitude")
    #if((k==3) & (j==0)):a1.set_ylim(-0.2,0.6)
    #if((k==3) & (j==1)):a1.set_ylim(-0.12,0.4)
    a1.grid(True)
    a1.plot(time1, RFo, '-', color="grey",label='Observed')
    #a1.plot(time2[:626], RFp[:626], 'b-', label='Predicted')
    a1.plot(time2, RFp, 'b-', label='Predicted')
    a1.legend()

    if(mtype ==0):d, beta, vpvs, qa, qb, pv, pd = v2mod(velmod)  # Convert velocity model from Voronoi format to plot format
    if(mtype ==1):pv, pd = l2mod(velmod)  # Convert velocity model from Layer format to plot format
    if(mtype ==2):pv, pd = d2mod(velmod)  # Convert velocity model from Layer format to plot format
    
    a0.set_title(" Velocity model")                   # Plot velocity model with Receiver function
    a0.set_xlabel('Vs (km/s)')
    a0.set_ylabel('Depth (km)')
    a0.plot(pv,pd,'g-')
    a0.set_xlim(vmin,vmax)
    a0.set_ylim(dmin,dmax)
    a0.invert_yaxis()
    if(mtype==0): a0.plot(velmod[:,1],velmod[:,0],ls="",marker="o",markerfacecolor='none', markeredgecolor='k')

    #plt.tight_layout()
    plt.show()
    plt.savefig('RF_mod_plot.pdf',format='PDF')
    return pv,pd
##################################################################################
if __name__ == "__main__":
    #print(sys.path)
    velmod0 = np.zeros([7,3]) # Set up velocity reference model
    velmod0[0] = [8.370596, 3.249075, 1.7]
    velmod0[1] = [17.23163, 3.001270, 1.7]
    velmod0[2] = [1.9126695E-02, 2.509443, 1.7]
    velmod0[3] = [19.78145, 3.562691, 1.7]
    velmod0[4] = [41.73066, 4.225965, 1.7]
    velmod0[5] = [14.35261, 2.963322, 1.7]
    velmod0[6] = [49.92358, 4.586726, 1.7]
    velmod = np.copy(velmod0)

	# plot waveforms

    #time1, RFo = np.loadtxt("RF_obs.dat", unpack=True) # Read in observed Receiver function
	
    #time2, RFp = RF.rfcalc(velmod)                     # Calculate predicted Receiver function using velocity model
    time1, RFo = rfcalc(velmod) # Read in observed Receiver function
    time2, RFp = rfcalc(velmod,sn=0.5,seed=61254557)                     # Calculate predicted Receiver function using velocity model
    plt.title(" Observed and predicted receiver functions")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.plot(time1, RFo, 'k-', label='Observed')
    plt.plot(time2[:626], RFp[:626], 'r-', label='Predicted')
    plt.legend()
    plt.show()
    plt.savefig('RF_plot.pdf',format='PDF')

	# Set up data Covariance matrix
	# 2.5, 0.01 = correlation half-width and amplitude of Gaussian noise
    ndata = len(RFp)
    Cdinv = InvDataCov(2.5,0.01,ndata) 

# Calculate waveform misfit for reference model

    res = RFo-RFp
    mref = 	np.dot(res,np.transpose(np.dot(Cdinv, res)))/2.0
    print (' Waveform misfit of reference model',mref)
