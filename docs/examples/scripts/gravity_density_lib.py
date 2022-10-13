"""Auxiliary code for gravity density model

Adapted from inversion-test-problems: 
https://github.com/inlab-geo/inversion-test-problems/blob/main/contrib/gravityforward/__init__.py

Author: Hannes Hollmann


The file name should end with "_lib.py", otherwise our bot may fail when generating
scripts for Sphinx Gallery. Furthermore, we recommend the file name to start with your
forward problem name, to align well with the naming of Jupyter notebook.

"""

import numpy as np
import matplotlib.pyplot as plt

def kernel(ii,jj,kk,dx,dy,dz,dim):
    
    r = (dx[:, ii] ** 2 + dy[:, jj] ** 2 + dz[:, kk]** 2) ** (0.50)

    dz_r = dz[:, kk] + r
    dy_r = dy[:, jj] + r
    dx_r = dx[:, ii] + r

    dxr = dx[:, ii] * r
    dyr = dy[:, jj] * r
    dzr = dz[:, kk] * r
    
    dydz = dy[:, jj] * dz[:, kk]
    dxdy = dx[:, ii] * dy[:, jj]
    dxdz = dx[:, ii] * dz[:, kk]
    
    if dim=="gx":
        g = (-1) ** (ii + jj + kk) * (dy[:, jj] * np.log(dz_r) + dz[:, kk]* np.log(dy_r) - dx[:, ii] * np.arctan(dydz / dxr))
    elif dim=="gy":
        g = (-1) ** (ii + jj + kk) * (dx[:, ii] * np.log(dz_r) + dz[:, kk]* np.log(dx_r) - dy[:, jj] * np.arctan(dxdz / dyr))
    elif dim=="gz":
        g = (-1) ** (ii + jj + kk) * (dx[:, ii] * np.log(dy_r) + dy[:, jj] * np.log(dx_r) - dz[:, kk]* np.arctan(dxdy / dzr))
    elif dim=="gxx":
        arg = dy[:, jj] * dz[:, kk] / dxr
        # It said g-= ... - maybe neet to switch vorzeichen?
        g = ((-1) ** (ii + jj + kk) * (dxdy / (r * dz_r)+ dxdz / (r * dy_r)- np.arctan(arg)+ dx[:, ii]* (1.0 / (1 + arg ** 2.0))* dydz/ dxr ** 2.0* (r + dx[:, ii] ** 2.0 / r)))
    elif dim=="gxy":
        arg = dy[:, jj] * dz[:, kk] / dxr
        g = ((-1) ** (ii + jj + kk) * (np.log(dz_r)+ dy[:, jj] ** 2.0 / (r * dz_r)+ dz[:, kk] / r- 1.0/ (1 + arg ** 2.0)* (dz[:, kk] / r ** 2)* (r - dy[:, jj] ** 2.0 / r)))
    elif dim=="gxz":
        arg = dy[:, jj] * dz[:, kk] / dxr
        g = ((-1) ** (ii + jj + kk) * (np.log(dy_r)+ dz[:, kk] ** 2.0 / (r * dy_r)+ dy[:, jj] / r- 1.0/ (1 + arg ** 2.0)* (dy[:, jj] / (r ** 2))* (r - dz[:, kk] ** 2.0 / r)))
    elif dim=="gyy":
        arg = dx[:, ii] * dz[:, kk] / dyr
        g = ((-1) ** (ii + jj + kk) * (dxdy / (r * dz_r)+ dydz / (r * dx_r)- np.arctan(arg)+ dy[:, jj]* (1.0 / (1 + arg ** 2.0))* dxdz/ dyr ** 2.0* (r + dy[:, jj] ** 2.0 / r)))
    elif dim=="gyz":
        arg = dx[:, ii] * dz[:, kk] / dyr
        g = ((-1) ** (ii + jj + kk) * (np.log(dx_r)+ dz[:, kk] ** 2.0 / (r * (dx_r))+ dx[:, ii] / r- 1.0/ (1 + arg ** 2.0)* (dx[:, ii] / (r ** 2))* (r - dz[:, kk] ** 2.0 / r)))
    elif dim=="gzz":
        arg = dy[:, jj] * dz[:, kk] / dxr
        gxx = ((-1) ** (ii + jj + kk) * (dxdy / (r * dz_r)+ dxdz / (r * dy_r)- np.arctan(arg)+ dx[:, ii]* (1.0 / (1 + arg ** 2.0))* dydz/ dxr ** 2.0* (r + dx[:, ii] ** 2.0 / r)))
        arg = dx[:, ii] * dz[:, kk] / dyr
        gyy = ((-1) ** (ii + jj + kk) * (dxdy / (r * dz_r)+ dydz / (r * dx_r)- np.arctan(arg)+ dy[:, jj]* (1.0 / (1 + arg ** 2.0))* dxdz/ dyr ** 2.0* (r + dy[:, jj] ** 2.0 / r)))
        g=-gxx-gyy
    return g

def calculate_gravity(model, x_final, y_final, z_final, recvec, jacobian=False, return_both=False):
    if return_both==True:
        jacobian=True
    from scipy.constants import G as G    
    # Tolerance implementation follows SimPEG, discussed in Nagy et al., 2000
    tol=1e-4

    gx_rec=np.zeros(len(recvec))
    gy_rec=np.zeros(len(recvec))
    gz_rec=np.zeros(len(recvec))
    
    if jacobian==True:
        Jx_rec=np.zeros([len(recvec),len(x_final)])
        Jy_rec=np.zeros([len(recvec),len(x_final)])
        Jz_rec=np.zeros([len(recvec),len(x_final)])

    
    for recno in range(len(recvec)):

        dx=x_final-recvec[recno,0]
        dy=y_final-recvec[recno,1]
        dz=z_final-recvec[recno,2]

        min_x=np.min(np.diff(dx))
        min_y=np.min(np.diff(dy))
        min_z=np.min(np.diff(dz))

        dx[np.abs(dx) / min_x < tol] = tol * min_x
        dy[np.abs(dy) / min_y < tol] = tol * min_y
        dz[np.abs(dz) / min_z < tol] = tol * min_z

        Jx=0
        Jy=0
        Jz=0

        for ii in range(2):
            for jj in range(2):
                for kk in range(2):

                    # gx, gy apppear to work, but need confcirmation.
                    # gz is tested and works
                    Jx+=kernel(ii,jj,kk,dx,dy,dz,"gx")
                    Jy+=kernel(ii,jj,kk,dx,dy,dz,"gy")
                    Jz+=kernel(ii,jj,kk,dx,dy,dz,"gz")
                        
        
        # Multiply J (Nx1) with the model density (Nx1) element-wise
        gx_rec[recno] = 1e5*G*sum(model*Jx)
        gy_rec[recno] = 1e5*G*sum(model*Jy)
        gz_rec[recno] = 1e5*G*sum(model*Jz)
        
        if jacobian==True:
            Jx_rec[recno,:] = Jx
            Jy_rec[recno,:] = Jy
            Jz_rec[recno,:] = Jz
            
    if jacobian==False:    
        return gx_rec, gy_rec, gz_rec
    elif return_both==True:
        return gx_rec, gy_rec, gz_rec, Jx_rec,  Jy_rec,  Jz_rec
    else: 
        return Jx_rec,  Jy_rec,  Jz_rec
    
def calculate_gradiometry(model, x_final, y_final, z_final, recvec):

    # Tolerance implementation follows SimPEG, discussed in Nagy et al., 2000
    tol=1e-4

    gxx_rec=np.zeros(len(recvec))
    gxy_rec=np.zeros(len(recvec))
    gxz_rec=np.zeros(len(recvec))
    gyy_rec=np.zeros(len(recvec))
    gyz_rec=np.zeros(len(recvec))
    gzz_rec=np.zeros(len(recvec))

    for recno in range(len(recvec)):

        dx=x_final-recvec[recno,0]
        dy=y_final-recvec[recno,1]
        dz=z_final-recvec[recno,2]

        min_x=np.min(np.diff(dx))
        min_y=np.min(np.diff(dy))
        min_z=np.min(np.diff(dz))

        dx[np.abs(dx) / min_x < tol] = tol * min_x
        dy[np.abs(dy) / min_y < tol] = tol * min_y
        dz[np.abs(dz) / min_z < tol] = tol * min_z

        Jxx=0
        Jxy=0
        Jxz=0
        Jyy=0
        Jyz=0
        Jzz=0
        
        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    
                    # Gradiometry kernels NOT checked yet. Just look good.
                    Jxx+=kernel(ii,jj,kk,dx,dy,dz,"gxx")
                    Jxy+=kernel(ii,jj,kk,dx,dy,dz,"gxy")
                    Jxz+=kernel(ii,jj,kk,dx,dy,dz,"gxz")
                    Jyy+=kernel(ii,jj,kk,dx,dy,dz,"gyy")
                    Jyz+=kernel(ii,jj,kk,dx,dy,dz,"gyz")
                    Jzz+=kernel(ii,jj,kk,dx,dy,dz,"gzz")

                    
        # Multiply J (Nx1) with the model density (Nx1) element-wise
        gxx_rec[recno] = 1e8*G*sum(model*Jxx)
        gxy_rec[recno] = 1e8*G*sum(model*Jxy)
        gxz_rec[recno] = 1e8*G*sum(model*Jxz)
        gyy_rec[recno] = 1e8*G*sum(model*Jyy)
        gyz_rec[recno] = 1e8*G*sum(model*Jyz)
        gzz_rec[recno] = 1e8*G*sum(model*Jzz)
    
    
    return gxx_rec, gxy_rec, gxz_rec, gyy_rec, gyz_rec, gzz_rec 


# -----------------------------------------------------
# Utility loading functions

def load(file_name):
    data_source = np.DataSource()
    url_base = "https://github.com/inlab-geo/cofi-examples/raw/main/notebooks/gravity"

    if data_source.exists(file_name):
        tmp = np.load(file_name)
    else:
        abs_path = data_source.abspath(f"{url_base}/{file_name}")
        if data_source.exists(abs_path):
            tmp = np.load(abs_path)
        else:
            print(f"Downloading models {file_name}...")
            data_source.open(f"{url_base}/{file_name}")
            tmp = np.load(abs_path)
    return tmp


def load_gravity_model():
    # Load density true model
    tmp = load("gravity_model.npz")

    # Extract density model
    model=tmp['model']
    x_nodes = tmp['x_nodes']
    y_nodes = tmp['y_nodes']
    z_nodes=tmp['z_nodes']
    rec_coords=tmp['rec_coords']

    return rec_coords, x_nodes, y_nodes, z_nodes, model


def load_starting_models():
    # Load density true model & starting models
    tmp = load("gravity_starting_models.npz")

    # Process starting models
    Starting_model1=tmp['starting_model1']
    Starting_model1[Starting_model1 > 0 ]=10
    Starting_model2=tmp['starting_model2']
    Starting_model3 = np.zeros((12**3))

    return Starting_model1, Starting_model2, Starting_model3
    

# -----------------------------------------------------
# New functions, defined just for CoFI
# The ones above are from my forward code. 
# Only difference between these two is "jacobian=True", but necessary here to separate them

def get_forward_jacobian():
    # Load model geometry 
    rec_coords, x_nodes, y_nodes, z_nodes, _ = load_gravity_model()

    def forward(model):
        gx, gy, gz = calculate_gravity(model,x_nodes,y_nodes,z_nodes,rec_coords)
        return gz

    def get_jacobian(model):
        Jx, Jy, Jz = calculate_gravity(model,x_nodes,y_nodes,z_nodes,rec_coords, jacobian=True)
        return Jz

    return forward, get_jacobian

forward, get_jacobian = get_forward_jacobian()

def depth_weight(z,z0,beta):
    return ((-z[::-1]+z0)**(-beta/2))

def reg_l1(model, W):
    return np.linalg.norm(W @ model, 1)

def reg_l2(model, W):
    return np.linalg.norm(W @ model, 2)

def reg_gradient_l1(model, W):
    return W @ np.sign(model)

def reg_gradient_l2(model, W):
    return 2 * model.T @ W.T @ W
