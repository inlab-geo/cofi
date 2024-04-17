In an ensemble method we seek to sample the posterior distrbution which is the product of the prior distirbution and the likelihood function, with the likelihood given as:
$$
p(\mathbf{d}|\mathbf{m})=  \frac{1}{\sqrt{(2 \pi)^n |C_d|}} e^{-\frac{1}{2} (\mathbf{d} -\mathrm{f}(\mathbf{m}))^{\mathrm{T}} C_{d}^{-1}(\mathbf{d} -\mathrm{f}(\mathbf{m}))} 
$$
with for practical application the log likelihood will be used.

We use a uniform distribution as the prior.
