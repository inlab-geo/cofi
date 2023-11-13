import numpy as np

from . import BaseInferenceTool, error_handler


# Add CoFIBorderCollieOptimization class into src/cofi/tools/__init__.py
# FIXME 1. "from _cofi_border_collie_optimization import CoFIBorderCollieOptimization"
# FIXME 2. add "CoFIBorderCollieOptimization" to "__all__" list
# FIXME 3. add "CoFIBorderCollieOptimization" to "inference_tools_table" dictionary
# FIXME Remove above comments lines after completed

import random

class CoFIBorderCollieOptimization(BaseInferenceTool):
    r"""Implentation of a naive Border Collie Optimization

    T. Dutta, S. Bhattacharyya, S. Dey and J. Platos, "Border Collie Optimization," in IEEE Access, vol. 8, pp. 109177-109197, 2020, doi: 10.1109/ACCESS.2020.2999540

    FIXME Any extra information about the tool
    """
    documentation_links = []        # FIXME required
    short_description = (
        "CoFI's implemntation of Border Collie Optimization"
            )          # FIXME required

    @classmethod
    def required_in_problem(cls) -> set:        # FIXME implementation required
        return {"objective","bounds","model_shape"}
    
    @classmethod
    def optional_in_problem(cls) -> dict:       # FIXME implementation required
        return {"initial_model":[]}

    @classmethod
    def required_in_options(cls) -> set:        # FIXME implementation required
        return set()

    @classmethod
    def optional_in_options(cls) -> dict:       # FIXME implementation required
        return {"number_of_iterations":50,"flock_size":50,"seed":42}

    def __init__(self, inv_problem, inv_options):       # FIXME implementation required
        super().__init__(inv_problem, inv_options)
        self._components_used = list(self.required_in_problem())
        # self model size
        self.mod_size=np.prod(inv_problem.model_shape)
        # lower and upper boundaries
        for i in range(self.mod_size):
            self.lower_bounds=inv_problem.bounds[i][0]
            self.upper_bounds=inv_problem.bounds[i][1]
        self.initial_model=inv_problem.initial_model
        
        print(inv_problem.model_shape)


    def __call__(self) -> dict:                         # FIXME implementation required

        
        # number of iterations
        self.number_of_iterations = self._params["number_of_iterations"]
         # number of sheep - there are always three dogs
        self.pack_size=3
        self.flock_size = self._params["flock_size"]
        self.seed = self._params["seed"]
        pop,vel,acc,tim=self.initialise()

        for itr in range(self.number_of_iterations):
            fit,fit_max_val,fit_max_idx = self.fitness(pop)
            
            res = {
            "model": pop[max_idx,:],
        	}

            
            self.eye=0
            if itr==0:
                fit_opt=fit_max_val
            if fit_opt>fit_max_val:
                fit_fopt=fit_max_val

            self.fit_opt.append(fit_opt)

            if itr>0:
                if self.fit_opt[-1]>self.fit_opt[-2]:
                    k=k+1
                    if (k>5):
                        self.eye=0
                        k=0
            
            fit,pop,vel,acc,tim=self.herd(fit,pop,vel,acc,tim)
            vel_next,acc_next,tim_next=update_velocity_acceleration_and_time(fit,pop,vel,acc,tim)
            pop= update_positons(pop,vel,acc,tim)
            pop,acc,tim=check_positions
            
  
        return res
    
    @error_handler(
        when="FIXME (e.g. when solving / calling ...)",
        context="FIXME (e.g. in the process of solving / preparing)",
    )
    def _call_backend_tool(self):                       # FIXME implementation required
        raise NotImplementedError


    # The following are useful functions used by the main optimiser but it makes sense to keep them
    # seperate


    def initialise(self):
        pop = np.zeros([self.pack_size+self.flock_size,self.mod_size])
        print(self.mod_size)

        # if an intial model has been provided everyone starts there
        if self.initial_model!=[]:
            # there is probably a short hand way to do this...
            for i in range(self.pack_size+self.flock_size):
                pop[i,:] = self.initial_model
        else:
            pop=np.random.rand(self.pack_size+self.flock_size,self.mod_size)*(self.upper_bounds-self.lower_bounds)+self.lower_bounds
 
        vel = np.random.default_rng().uniform(size=(self.pack_size+self.flock_size,self.mod_size))
        acc = np.random.default_rng().uniform(size=(self.pack_size+self.flock_size,self.mod_size))
        tim = np.random.default_rng().uniform(size=(self.pack_size+self.flock_size,self.mod_size))

        self.fit_opt=[]


        return pop,vel,acc,tim


    def fitness(self,pop):

        fit=np.zeros([self.pack_size+self.flock_size,1])

        # could be done parallel
        for i in range(self.pack_size+self.flock_size):
            fit[i]=self.inv_problem.objective(pop[i,:])

        max_idx=np.argmin(fit)
        max_val=fit[max_idx]

        return fit,fit_max_val,fit_max_idx

    def herd(self,fit,pop,vel,acc,tim):
        fit1=fit.sort(axis=1)
        idx=fit.argsort(axis=1)

        pop_next= np.zeros([self.pack_size+self.flock_size,self.mod_size])   
        vel_next=np.zeros([self.pack_size+self.flock_size,self.mod_size])
        acc_next=np.zeros([self.pack_size+self.flock_size,self.mod_size])
        tim_next=np.zeros([self.pack_size+self.flock_size,1])
        fit_next=np.copy(fit)
        fit_next.sort()
        for j in range(np):
            pop_next[j,:]=pop[idx[j],:]
            vel_next[j,:]=v[idx[j],:]
            acc_next[j,:]=a[j,:]
            tim_next[j]=t[j]

        return fit_next,pop_next,vel_next,acc_next,tim_next

    def update_velocity_acceleration_and_time(self,fit,pop,vel,acc,tim):

        vel_next = np.zeros([self.pack_size+self.flock_size,self.mod_size])   
        vel_next=np.copy(vel)
        acc_next= np.zeros([self.pack_size+self.flock_size,self.mod_size])   
        tim_next= np.zeros([self.pack_size+self.flock_size,self.mod_size])   

        # tournament selection for choosing left and right dogs
        r1=np.random.randint(1,high=3)
        if r1==1:
            l1=2;
        else:
            l1=1; 

        # finding the dog values to choose which sheepe to gather and which to stalk
        f1=fit[0]
        f2=(fit[1]+fit[2])/2.0;
        f=0;

        if self.eye==1:
            if fit[r1]<fit[l1]:
                acc_next[l1,:]=(-1)*a[l1,:]
                f=l1;
            else:
                acc_next[r1,:]=(-1)*a[r1,:];
                f=r1;
        
        for i in range(self.pack_size+self.flock_size):
            for j in range(self.mod_size):
            #    velocity updates of dogs
                if (i<self.pack_size):
                    vel_next[i,j]=np.sqrt(vel[i,j]**2+2.0*acc[i,j]*pop[i,j])
                # velcoity update of sheep
                if (i>self.pack_size):
                    if (self.eye==1):
                        vel_next[i,j]=np.sqrt(vel[f,j]**2+2.0*acc[f,j]*pop[i,j])
                    else:
                        # Velocity updation of gathered sheep
                        if f1-fit[i]>f2-fit[i]:
                            vel_next[i,j]=np.sqrt(vel[1,j]**2+2.0*acc[1,j]*pop[i,j])

                        # Velocity updation of stalked sheep
                        if f1-fit[i]<=f2-fit[i]:
                            vel_next[i,j]=np.sqrt((v_next[1,j]*np.tan(np.random.randint(1,high=90))**2)+2.0*a[r1,j]*pop[r1,j])+np.sqrt((v_next[l1,j]*np.tan(np.random.randint(91,high=180))**2)+2.0*a[l1,j]*pop[l1,j])
                            vel_next[i,j]=vel_next[i,j]/2.0;
        # Updating of time and acceleration

        for i in range(self.pack_size+self.flock_size):
            s=0;
            for j in range(self.mod_size):
                acc_next[i,j]=np.abs(vel_next[i,j]-vel[i,j])/tim[i];
                s=s+(vel_next[i,j]-v[i,j]/acc_next[i,j]);
            tim_next[i]=np.abs(s);
    
        return vel_next,acc_next,tim_next

    def update_positons(self,pop,vel,acc,tim):
        pop_next= np.zeros([np,nd])   
        for i in range(np):
            for j in range(nd):
            # Updating the position of dogs
                if(i<=2):
                    pop_next[i,j]=vel[i,j]*tim[i]+(1./2.)*acc[i,j]*(tim[i]**2);
            # Updating position of sheep
                if(i>3):
                    if self.eye==1:
                        pop_next[i,j]=vel[i,j]*tim[i]-(1./2.)*acc[i,j]*(tim[i]**2);
                    else:
                        pop_next[i,j]=vel[i,j]*tim[i]+(1./2.)*acc[i,j]*(tim[i]**2);

        return pop_next

    def check_positions(self,pop,acc,tim):

        pop_next=np.copy(pop)
        acc_next=np.copy(acc)
        tim_next=np.copy(tim)
   
        for i in range(self.pack_size+self.flock_size):
            for j in range(self.mod_size):
                if pop[i,j]>=m_ub[j] or pop[i,j]<=m_lb[j] or pop[i,j]==0 :
                    pop_next[i,j]=np.random.rand(self.pack_size+self.flock_size,self.mod_size)*(self.upper_bounds-self.lower_bounds)+self.lower_bounds
                    acc_next[i,j]=np.random.rand()
                    tim_next[i]=np.random.rand()
                if np.isnan(acc[i,j]) or acc[i,j]==0:
                    pop_next[i,j]=np.random.rand(self.pack_size+self.flock_size,self.mod_size)*(self.upper_bounds-self.lower_bounds)+self.lower_bounds
                    acc_next[i,j]=np.random.rand()
                    t_next[i]=np.random.rand();
                if np.isnan(v[i,j]) or vel[i,j]==0:
                    pop_next[i,j]=np.random.rand(self.pack_size+self.flock_size,self.mod_size)*(self.upper_bounds-self.lower_bounds)+self.lower_bounds
                    acc_next[i,j]=np.random.rand()
                    tim_next[i]=np.random.rand();
                if np.isnan(t_next[i]) or tim_next[i]==0:
                    pop_next[i,j]=np.random.rand(self.pack_size+self.flock_size,self.mod_size)*(self.upper_bounds-self.lower_bounds)+self.lower_bounds
                    acc_next[i,j]=np.random.rand()
                    tim_next[i]=np.random.rand();
        return pop_next,acc_next,tim_next


