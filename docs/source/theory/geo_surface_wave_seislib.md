To map lateral variations in surface-wave velocity, SeisLib implements a least-squares inversion scheme based on ray theory. This method rests on the assumption that surface waves propagate, from a given point on the Earth’s surface to another, without deviating from the great-circle path connecting them. Under this assumption, the traveltime along the great-circle path can be written $t = \int_{\mathrm{path}}{s(\phi(l), \theta(l)) dl}$, 
where $\phi$ and $\theta$ denote longitude and latitude, and $s$ the sought Earth's slowness.

Let us consider a discrete parameterization of the Earth's surface, and assume each block 
(or grid cell) of such parameterization has constant slowness. The above integral expression 
can then be reformulated in the discrete form

$$
s = \frac{1}{L} \sum_{n}{s_n l_n},
$$

where $L$ is the length of the great-circle path and $l$ the distance traveled by the surface wave through the $n$th block. The above equation represents the *forward* calculation that allows for retrieving the average velocity of propagation between two points on the Earth's surface (i.e., the quantity which is typically measured in ambient-noise seismology), provided that the (discrete) spatial variations in velocity (or slowness) are known.

If we now define the $m \times n$ matrix such that $A_{ij} = \frac{l_j}{L_i}$, where $L_i$ is the length of the great circle associated with $i$th observation, we can switch to matrix notation and write

$$
{\bf A \cdot x} = {\bf d},
$$

where $\bf d$ is an $m$-vector whose $k$th element corresponds to the measured slowness, and $\bf x$ the sought $n$-vector whose $k$th element corresponds to the model coefficient $s_k$. Matrix $\bf A$, also known as "data kernel" or "Jacobian", is computed numerically in a relatively simple fashion. For each pair of receivers for which a velocity measurement is available, its $i$th entries is found by calculating the fraction of great-circle path connecting them through each of the $n$ blocks associated with the parameterization.

In geophysical applications, the system of linear equations (equation above) is usually ill-conditioned, meaning that it is not possible to find an exact solution for $\bf x$. (In our case, it is strongly overdetermined, i.e. $m \gg n$.) We overcome this issue by first assuming that the target slowness model is approximately known, i.e. ${\bf x}_0 \sim \bf{x}$. We then invert for the regularized least-squares solution

$$
{\bf x} = {\bf x}_0 + \left( {\bf A}^T \cdot {\bf A} + \mu^2 {\bf R}^T \cdot {\bf R} \right)^{-1} \cdot {\bf A}^T \cdot ({\bf d} - {\bf A} \cdot {\bf x}_0),
$$

where the roughness of the final model is determined by the scalar weight $\mu$ and the roughness operator $\bf R$ is dependent on the parameterization (for technical details on its computation, see [*Magrini et al. (2022)*](https://doi.org/10.1093/gji/ggac236)).
