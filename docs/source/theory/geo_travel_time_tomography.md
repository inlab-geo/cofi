The goal in travel-time tomography is to infer details about the velocity structure of a medium, given measurements of the minimum time taken for a wave to propagate from source to receiver. For seismic travel times, as we change our model, the route of the fastest path from source to receiver also changes. This makes the problem nonlinear, as raypaths also depend on the sought after velocity or slowness model. 

Provided the 'true' velocity structure is not *too* dissimilar from our initial guess, travel-time tomography can be treated as a weakly non-linear problem. In this notebook we optionally treat the ray paths as fixed, and so it becomes a linear problem, or calculate rays in the velociuty model.

The travel-time of an individual ray can be computed as $$t = \int_\mathrm{path} \frac{1}{v(\mathbf{x})}\,\mathrm{d}\mathbf{x}$$
This points to an additional complication: even for a fixed path, the relationship between velocities and observations is not linear. However, if we define the 'slowness' to be the inverse of velocity, $s(\mathbf{x}) = v^{-1}(\mathbf{x})$, we can write
$$t = \int_\mathrm{path} {s(\mathbf{x})}\,\mathrm{d}\mathbf{x}$$
which *is* linear.


We will assume that the object we are interested in is 2-dimensional slowness field. If we discretize this model, with $N_x$ cells in the $x$-direction and $N_y$ cells in the $y$-direction, we can express $s(\mathbf{x})$ as an $N_x \times N_y$ vector $\boldsymbol{s}$. 

**For the linear case**, this is related to the data by
$$d_i = A_{ij}s_j $$
where $d_i$ is the travel time of the ith path, and where $A_{ij}$ represents the path length in cell $j$ of the discretized model.

**For the nonlinear case**, this is related to the data by
$$\delta d_i = A_{ij}\delta s_j $$
where $\delta d_i$ is the difference in travel time, of the ith path, between the observed time and the travel time in the reference model. Here $A_{ij}$ represents the path length in cell $j$ of the discretized model. The parameters $\delta s_j$ are slowness perturbations to the reference model.
