#!/usr/bin/env python
# coding: utf-8

# # 1. CoFI Basic Concepts
# 
# In this tutorial, we get to know the concepts of "objective" and "solver" in the context of CoFI (Common Framework of Inversion). 
# 
# The core idea of CoFI is to provide a standard way to define both geophysical problems and inverse solvers, so that different problems (from different sources) can be fed into various solvers that implement this interface.
# 
# Therefore, knowing how to define problems (we name it "objective") and solvers is the first step in this tutorial.
# 
# The following two blocks are what they look like and what they have:

# In[1]:


class BaseObjective:
    def __init__(self):
        pass

    def misfit(self, model):
        pass


# In[2]:


class BaseSolver:
    def __init__(self, objective):
        pass
    
    def solve(self):
        pass


# The basic requirement of an "objective" is to provide an evaluation given a model - we call this function "misfit", so that we want it to be as small as possible.
# 
# The basic requirement of a "solver" is to find information about the best-fit model or model distribution. Given a problem description (in form of an "objective" instance), each solver has its own way but all will have the function name "solve" to start the process and return a result.
# 
# Note that both of the classes defined above are a part of CoFI package already, so you don't need to define a "BaseObjective" or "BaseSolver". Whenever we want to define a new problem, we define a subclass of "BaseObjective", like how we'll do it in the next code block below. The same idea applies to "BaseSolver".

# ### Example: an "objective"
# 
# Here's a minimum example! Let's see how we can define an "objective" if we'd like our solver to calculate the "$x$" that minimises "$x^2-2x+1$". So here the value of this function is our "misfit".

# In[3]:


class MyOwnObjective(BaseObjective):
    def __init__(self):
        pass

    def misfit(self, model):
        return model * model - 2 * model + 1
    


# In practice, we usually use collected data to calculate the misfit. In such cases, you can pass in your dataset through the `__init__` constructor and then use them to calculate the misfit.

# ### Example: a "solver"

# Now that we've got a well defined "objective", let's define a very simple solver. 

# In[4]:


class MyOwnSolver(BaseSolver):
    def __init__(self, objective):
        self.objective = objective

    def solve(self):
        test_range = [0,1,2,3]
        test_misfits = []
        for test_model in test_range:
            misfit = self.objective.misfit(test_model)
            test_misfits.append(misfit)
        min_misfit = min(test_misfits)
        min_index = test_misfits.index(min_misfit)
        return test_range[min_index]


# This is totally a dumb solver, which only compare the misfit value within an unreasonably small range and return the one that has the lowest misfit. In reality, the "solve" function can be very complicated or wrap another optimisation or sampling method. 

# ### Example: using `MyOwnSolver` to solve `MyOwnObjective`
# 
# An example of using the classes defined above:

# In[5]:


objective = MyOwnObjective()
solver = MyOwnSolver(objective)
model = solver.solve()
print("best fit model according to our solver:", model)


# ### Under "objective": "forward"
# 
# Here we are talking about forward operators, as another possible option to define an objective. 
# 
# Let's take the cannon ball as an example, where there are 3 unknowns:
# - $m_1$, starting height above the surface
# - $m_2$, initial velocity
# - $m_3$, gravitational acceleration
# 
# With $m_1$, $m_2$ and $m_3$, we get the following relationship between position and time:
# $$ y(t) = m_1 + m_2 t - \frac{1}{2} m_3 t^2 $$
# 
# It's common to use data collected from an experiment to estimate these unknown parameters. If we define this problem by subclassing "BaseObjective":

# In[6]:


class CannonBallObjective(BaseObjective):
    def __init__(self, t, y):
        self.t = t
        self.y = y

    def misfit(self, model):
        y_predicted = model[0] + model[1] * self.t - 1/2 * model[2] * self.t * self.t
        residual = y_predicted - self.y
        return residual.T @ residual


# Given a forward problem, if we'd like to use least squares to calculate misfit, then the formula is always to perform a forward operation and then calculate the L2 norm of the residual vector. So I came to thinking maybe we can let users define less if they want to, and implement this pattern on our side.

# In[7]:


# LeastSquaresObjective is included in cofi, here (below) is only some sample code
# from cofi.cofi_objective import LeastSquareObjective
class LeastSquareObjective(BaseObjective):
    def __init__(self, x, y, forward):
        self.x = x
        self.y = y
        self.forward = forward

    def misfit(self, model):
        y_predicted = self.forward(model, self.x)
        residual = y_predicted - self.y
        return residual.T @ residual


# The above `LeastSquareObjective` is also included in `cofi` so you don't have to define it by yourself. The convenience of it is that users only need to define a forward operator in the form of either a function or a class (namely a subclass of `BaseForward` class).
# 
# See the following example, in which the cannon ball forward operator is defined and passed into `LeastSquareObjective`.

# In[8]:


# define forward operator
def cannon_ball_forward(model, t):
    return model[0] + model[1] * t - 1/2 * model[2] * t * t

# sample experiment data
data_y = [26.94, 33.45, 40.72, 42.32, 44.30, 47.19, 43.33, 40.13]
t = [1,2,3,4,5,6,7,8]

# instantiate objective
cannon_ball_obj = LeastSquareObjective(t, data_y, cannon_ball_forward)

# now cannon_ball_obj can be passed to solvers...
# scipy_solver = cofi.optimizers.ScipyOptimizerSolver(cannon_ball_obj)
# model = scipy_solver.solve()


# In the X-Ray Tomography and Receiver Function examples (in section 3 and 4), there are both "forward" class and "objective" class relating to them, and the "objective" class typically contains an instance of "forward" as a field. I'm imagining that "objective" class, as a problem statement, can contain some user defined metadata (like the pixel dimension of X-Ray Tomography problem).
# 
# If this "forward" idea still sounds confusing, maybe trying out the `ReceiverFunction` section 4 (in step 2) will give you a better idea.
# 
# When we utilise test problems from `inversion-test-suite`, implementing a forward operator and then building an objective class from it can be more convenient. And forward operators defined explicitly can also help you generate sample data, etc.

# ----
