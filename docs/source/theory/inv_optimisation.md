The objective function we are minimizing with parameter estimation is given as:

$$
\Psi(\mathbf{m}) = (\mathbf{d} -\mathrm{f}(\mathbf{m}))^{\mathrm{T}} C_{d}^{-1}(\mathbf{d} -\mathrm{f}(\mathbf{m})) + \lambda \mathbf{m}^{T} W^{\mathrm{T}} W \mathbf{{m}},
$$

where $\mathbf{d}$ represents the data vector of measured phase velocities, $\mathrm{f}(\mathbf{m})$ is the model prediction, $C_d^{-1}$ is the inverse of the data covariance matrix, $W$ the damping matrix, $\mathbf{m}$ the model vector and $\lambda$ a regularization factor. The model update is then given as

$$
\begin{equation} \Delta \mathbf{m}= (\underbrace{\mathbf{J}^T \mathbf{C}_d^{-1} \mathbf{J}+\lambda W^{T} W}_{\mathbf{Hessian}})^{-1}
(\underbrace{ \mathbf{J}^T\mathbf{C}_d^{-1} 
(\mathbf{d}-\mathrm{f}(\mathbf{m}))+\lambda W^{T} W \mathbf{m}}_{\mathbf{Gradient}}),
\end{equation} 
$$

where $J$ represents the Jacobian.
