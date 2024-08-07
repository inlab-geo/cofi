
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "tutorials/generated/travel_time_tomography.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_tutorials_generated_travel_time_tomography.py>`
        to download the full example code

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_tutorials_generated_travel_time_tomography.py:


Linear & non-linear travel time tomography
==========================================

.. GENERATED FROM PYTHON SOURCE LINES 9-14

|Open In Colab|

.. |Open In Colab| image:: https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670
   :target: https://colab.research.google.com/github/inlab-geo/cofi-examples/blob/main/tutorials/2_travel_time_tomography.ipynb


.. GENERATED FROM PYTHON SOURCE LINES 17-29

--------------

What we do in this notebook
---------------------------

Here we apply CoFI to two geophysical examples:

-  a **linear seismic travel time tomography** problem
-  a **nonlinear travel time tomography** cross borehole problem

--------------


.. GENERATED FROM PYTHON SOURCE LINES 32-42

Learning outcomes
-----------------

-  A demonstration of running CoFI for a regularized linear parameter
   estimation problem. Can be used as an example of a CoFI **template**.
-  A demonstration of how a (3rd party) nonlinear forward model can be
   imported from geo-espresso and used. Fast Marching algorithm for
   first arriving raypaths.
-  See how nonlinear iterative matrix solvers can be accessed in CoFI.


.. GENERATED FROM PYTHON SOURCE LINES 42-47

.. code-block:: Python


    # Environment setup (uncomment code below)

    # !pip install -U cofi geo-espresso








.. GENERATED FROM PYTHON SOURCE LINES 52-119

Problem description
-------------------

The goal in **travel-time tomography** is to infer details about the
velocity structure of a medium, given measurements of the minimum time
taken for a wave to propagate from source to receiver.

At first glance, this may seem rather similar to the X-ray tomography
problem. However, there is an added complication: as we change our
model, the route of the fastest path from source to receiver also
changes. Thus, every update we apply to the model will inevitably be (in
some sense) based on incorrect assumptions.

Provided the ‘true’ velocity structure is not *too* dissimilar from our
initial guess, travel-time tomography can be treated as a weakly
non-linear problem.

In this notebook, we illustrate both linear and one non-linear
tomography.

In the first example the straight ray paths are fixed and independent of
the medium through which they pass. This would be the case for X-ray
tomography, where the data represent amplitude changes across the
medium, or seismic tomography under the fixed ray assumption, where the
data represent travel times across the medium.

In the second example we iteratively update seismic travel times and ray
paths as the seismic velocity model changes, which creates a nonlinear
tomographic problem.

In the seismic case, the travel-time of an individual ray can be
computed as

.. math:: t = \int_\mathrm{path} \frac{1}{v(\mathbf{x})}\,\mathrm{d}\mathbf{x}

This points to an additional complication: even for a fixed path, the
relationship between velocities and observations is not linear. However,
if we define the ‘slowness’ to be the inverse of velocity,
:math:`s(\mathbf{x}) = v^{-1}(\mathbf{x})`, we can write

.. math:: t = \int_\mathrm{path} {s(\mathbf{x})}\,\mathrm{d}\mathbf{x}

which *is* linear.

We will assume that the object we are interested in is 2-dimensional
slowness field. If we discretize this model, with :math:`N_x` cells in
the :math:`x`-direction and :math:`N_y` cells in the
:math:`y`-direction, we can express :math:`s(\mathbf{x})` as an
:math:`N_x \times N_y` vector :math:`\boldsymbol{s}`.

**For the linear case**, this is related to the data by

.. math:: d_i = A_{ij}s_j 

where :math:`d_i` is the travel time of the :math:`i` th path, and where
:math:`A_{ij}` represents the path length of raypath :math:`i` in cell
:math:`j` of the discretized model.

**For the nonlinear case**, this is related to the data by

.. math:: \delta d_i = A_{ij}\delta s_j 

where :math:`\delta d_i` is the difference in travel time, of the
:math:`i` th path, between the observed time and the travel time in the
reference model, and the parameters :math:`\delta s_j` are slowness
perturbations to the reference model.


.. GENERATED FROM PYTHON SOURCE LINES 119-126

.. code-block:: Python


    import numpy as np
    import matplotlib.pyplot as plt

    import cofi
    import espresso








.. GENERATED FROM PYTHON SOURCE LINES 131-134

1. Linear Travel Time Tomography
--------------------------------


.. GENERATED FROM PYTHON SOURCE LINES 137-140

To illustrate the setting we plot a reference model supplied through the
*espresso* Xray example, together with 100 raypaths in the dataset.


.. GENERATED FROM PYTHON SOURCE LINES 140-143

.. code-block:: Python


    linear_tomo_example = espresso.XrayTomography()








.. GENERATED FROM PYTHON SOURCE LINES 145-153

.. code-block:: Python


    # linear_tomo_example.plot_model(linear_tomo_example.good_model, paths=True);
    # linear_tomo_example.plot_model(linear_tomo_example.good_model);
    plt.plot(0.5, 0.5, marker="$?$", markersize=130)
    for p in linear_tomo_example._paths[:100]:
         plt.plot([p[0],p[2]],[p[1],p[3]],'y',linewidth=0.5)
    print(' Data set contains ',len(linear_tomo_example._paths),' ray paths')




.. image-sg:: /tutorials/generated/images/sphx_glr_travel_time_tomography_001.png
   :alt: travel time tomography
   :srcset: /tutorials/generated/images/sphx_glr_travel_time_tomography_001.png
   :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 .. code-block:: none

     Data set contains  10416  ray paths




.. GENERATED FROM PYTHON SOURCE LINES 158-161

Step 1. Define CoFI ``BaseProblem``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. GENERATED FROM PYTHON SOURCE LINES 164-168

Now we: - set up the BaseProblem in CoFI, - supply it the data vector
from espresso example, (i.e. the :math:`\mathbf{d}` vector) - supply it
the Jacobian of the linear system (i.e. the :math:`A` matrix)


.. GENERATED FROM PYTHON SOURCE LINES 168-176

.. code-block:: Python


    linear_tomo_problem = cofi.BaseProblem()
    linear_tomo_problem.set_data(linear_tomo_example.data)
    linear_tomo_problem.set_jacobian(linear_tomo_example.jacobian(linear_tomo_example.starting_model)) # supply matrix A
    sigma = 0.1 # set noise level of data
    data_cov_inv = np.identity(linear_tomo_example.data_size) * (1/sigma**2)
    linear_tomo_problem.set_data_covariance_inv(data_cov_inv)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    Evaluating paths:   0%|          | 0/10416 [00:00<?, ?it/s]    Evaluating paths:   8%|▊         | 883/10416 [00:00<00:01, 8829.76it/s]    Evaluating paths:  17%|█▋        | 1795/10416 [00:00<00:00, 8996.20it/s]    Evaluating paths:  26%|██▌       | 2695/10416 [00:00<00:00, 8928.29it/s]    Evaluating paths:  34%|███▍      | 3590/10416 [00:00<00:00, 8935.83it/s]    Evaluating paths:  43%|████▎     | 4491/10416 [00:00<00:00, 8961.00it/s]    Evaluating paths:  52%|█████▏    | 5388/10416 [00:00<00:00, 8918.97it/s]    Evaluating paths:  60%|██████    | 6301/10416 [00:00<00:00, 8985.76it/s]    Evaluating paths:  69%|██████▉   | 7200/10416 [00:00<00:00, 8948.43it/s]    Evaluating paths:  78%|███████▊  | 8095/10416 [00:00<00:00, 8905.91it/s]    Evaluating paths:  86%|████████▋ | 8997/10416 [00:01<00:00, 8939.01it/s]    Evaluating paths:  95%|█████████▍| 9891/10416 [00:01<00:00, 8920.57it/s]    Evaluating paths: 100%|██████████| 10416/10416 [00:01<00:00, 8959.48it/s]




.. GENERATED FROM PYTHON SOURCE LINES 181-194

Since :math:`\mathbf{d}` and :math:`G` have been defined then this
implies a linear system. Now we choose to regularize the linear system
and solve the problem

.. math::  \min_{\mathbf s} \phi({\mathbf d},{\mathbf s}) = ({\mathbf d} - A {\mathbf s})^T C_d^{-1} ({\mathbf d} - A {\mathbf s})~ + ~ \lambda ~{\mathbf s}D^TD{\mathbf s}

The matrix system we are solving is

.. math::


   (\mathbf{A}^T \textbf{C}_d^{-1} \textbf{A} + \lambda \mathbf D^T\mathbf D) \textbf{s} = \textbf{A}^T \mathbf C_d^{-1} \textbf{d}


.. GENERATED FROM PYTHON SOURCE LINES 194-203

.. code-block:: Python


    # set up regularization
    lamda = 0.5   # choose regularization constant
    reg_damping = lamda * cofi.utils.QuadraticReg(
        model_shape=(linear_tomo_example.model_size,)
    )
    linear_tomo_problem.set_regularization(reg_damping)
    print('Number of slowness parameters to be solved for = ',linear_tomo_example.model_size)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    Number of slowness parameters to be solved for =  2500




.. GENERATED FROM PYTHON SOURCE LINES 208-210

and lets print a summary of the set up.


.. GENERATED FROM PYTHON SOURCE LINES 210-213

.. code-block:: Python


    linear_tomo_problem.summary()





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    =====================================================================
    Summary for inversion problem: BaseProblem
    =====================================================================
    Model shape: Unknown
    ---------------------------------------------------------------------
    List of functions/properties set by you:
    ['jacobian', 'regularization', 'data', 'data_covariance_inv']
    ---------------------------------------------------------------------
    List of functions/properties created based on what you have provided:
    ['jacobian_times_vector']
    ---------------------------------------------------------------------
    List of functions/properties that can be further set for the problem:
    ( not all of these may be relevant to your inversion workflow )
    ['objective', 'log_posterior', 'log_posterior_with_blobs', 'log_likelihood', 'log_prior', 'gradient', 'hessian', 'hessian_times_vector', 'residual', 'jacobian_times_vector', 'data_misfit', 'regularization_matrix', 'forward', 'data_covariance', 'initial_model', 'model_shape', 'blobs_dtype', 'bounds', 'constraints']




.. GENERATED FROM PYTHON SOURCE LINES 218-221

Step 2. Define CoFI ``InversionOptions``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. GENERATED FROM PYTHON SOURCE LINES 224-227

Here we choose the backend tool for solving the tomographic system,
which is scipy’s least squares solver.


.. GENERATED FROM PYTHON SOURCE LINES 227-231

.. code-block:: Python


    tomo_options = cofi.InversionOptions()
    tomo_options.set_tool("scipy.linalg.lstsq")








.. GENERATED FROM PYTHON SOURCE LINES 236-239

Step 3. Define CoFI ``Inversion`` and run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. GENERATED FROM PYTHON SOURCE LINES 239-244

.. code-block:: Python


    tomo_inv = cofi.Inversion(linear_tomo_problem, tomo_options)
    tomo_inv_result = tomo_inv.run()
    tomo_inv_result.summary()





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    ============================
    Summary for inversion result
    ============================
    SUCCESS
    ----------------------------
    model: [1.13306453 0.86363911 1.01958229 ... 1.01319821 0.8615539  1.14691342]
    sum_of_squared_residuals: []
    effective_rank: 2500
    singular_values: [373.05549274 344.05222637 344.05222637 ...   1.4576611    1.35184016
       1.35184016]
    model_covariance: [[ 1.86880217e-01 -9.69914246e-02 -1.15714682e-02 ...  6.47051363e-05
      -2.09495749e-05 -2.00817961e-04]
     [-9.69914246e-02  3.02828183e-01 -6.75690464e-02 ... -4.09130322e-04
       3.44626731e-04 -2.09495749e-05]
     [-1.15714682e-02 -6.75690464e-02  2.21952501e-01 ...  3.27488527e-04
      -4.09130322e-04  6.47051363e-05]
     ...
     [ 6.47051363e-05 -4.09130322e-04  3.27488527e-04 ...  2.21952501e-01
      -6.75690464e-02 -1.15714682e-02]
     [-2.09495749e-05  3.44626731e-04 -4.09130322e-04 ... -6.75690464e-02
       3.02828183e-01 -9.69914246e-02]
     [-2.00817961e-04 -2.09495749e-05  6.47051363e-05 ... -1.15714682e-02
      -9.69914246e-02  1.86880217e-01]]




.. GENERATED FROM PYTHON SOURCE LINES 249-251

Lets plot the image to see what we got.


.. GENERATED FROM PYTHON SOURCE LINES 251-254

.. code-block:: Python


    ax = linear_tomo_example.plot_model(tomo_inv_result.model);




.. image-sg:: /tutorials/generated/images/sphx_glr_travel_time_tomography_002.png
   :alt: travel time tomography
   :srcset: /tutorials/generated/images/sphx_glr_travel_time_tomography_002.png
   :class: sphx-glr-single-img





.. GENERATED FROM PYTHON SOURCE LINES 259-302

Challenge: Fewer ray paths for linear travel time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Try and construct a tomographic solution with **fewer ray paths**.

Here we use 10416 ray paths with indices 0,10415. Try a different range
and see what you get.

How many ray paths do you need before the image becomes recognizable?

|Upload to Jamboard 1|

Start from the code template below:

::

   # data range
   idx_from, idx_to = (<CHANGE ME>, <CHANGE ME>)

   # basic settings
   d = linear_tomo_example.data
   G = linear_tomo_example.jacobian(linear_tomo_example.starting_model)

   # now attach all the info to a BaseProblem object
   mytomo = cofi.BaseProblem()
   mytomo.set_data(d[idx_from:idx_to])
   mytomo.set_jacobian(G[idx_from:idx_to,:])

   # run your problem (with the same InversionOptions) again
   mytomo_inv = cofi.Inversion(mytomo, tomo_options)
   mytomo_result = mytomo_inv.run()

   # check result
   fig = linear_tomo_example.plot_model(mytomo_result.model)
   plt.title(f'Recovered model from range ({idx_from}, {idx_to})')
   plt.figure()
   plt.title(' Raypaths')
   for p in linear_tomo_example._paths[idx_from:idx_to]:
       plt.plot([p[0],p[2]],[p[1],p[3]],'y',linewidth=0.05)

.. |Upload to Jamboard 1| image:: https://img.shields.io/badge/Click%20&%20upload%20your%20results%20to-Jamboard-lightgrey?logo=jamboard&style=for-the-badge&color=fcbf49&labelColor=edede9
   :target: https://jamboard.google.com/d/15UiYLe84zlkgLmi_ssbGuxRKyU-s4XuHSHsL8VppKJs/edit?usp=sharing


.. GENERATED FROM PYTHON SOURCE LINES 302-307

.. code-block:: Python


    # Copy the template above, Replace <CHANGE ME> with your answer










.. GENERATED FROM PYTHON SOURCE LINES 309-336

.. code-block:: Python


    #@title Solution

    # data range
    idx_from, idx_to = (0, 3000)                    # TODO try a different range

    # basic settings
    d = linear_tomo_example.data
    G = linear_tomo_example.jacobian(linear_tomo_example.starting_model)

    # now attach all the info to a BaseProblem object
    mytomo = cofi.BaseProblem()
    mytomo.set_data(d[idx_from:idx_to])
    mytomo.set_jacobian(G[idx_from:idx_to,:])

    # run your problem (with the same InversionOptions) again
    mytomo_inv = cofi.Inversion(mytomo, tomo_options)
    mytomo_result = mytomo_inv.run()

    # check result
    fig = linear_tomo_example.plot_model(mytomo_result.model)
    plt.title(f'Recovered model from range ({idx_from}, {idx_to})')
    plt.figure()
    plt.title(' Raypaths')
    for p in linear_tomo_example._paths[idx_from:idx_to]:
        plt.plot([p[0],p[2]],[p[1],p[3]],'y',linewidth=0.05)




.. rst-class:: sphx-glr-horizontal


    *

      .. image-sg:: /tutorials/generated/images/sphx_glr_travel_time_tomography_003.png
         :alt: Recovered model from range (0, 3000)
         :srcset: /tutorials/generated/images/sphx_glr_travel_time_tomography_003.png
         :class: sphx-glr-multi-img

    *

      .. image-sg:: /tutorials/generated/images/sphx_glr_travel_time_tomography_004.png
         :alt:  Raypaths
         :srcset: /tutorials/generated/images/sphx_glr_travel_time_tomography_004.png
         :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    Evaluating paths:   0%|          | 0/10416 [00:00<?, ?it/s]    Evaluating paths:   9%|▊         | 887/10416 [00:00<00:01, 8863.67it/s]    Evaluating paths:  17%|█▋        | 1791/10416 [00:00<00:00, 8962.44it/s]    Evaluating paths:  26%|██▌       | 2688/10416 [00:00<00:00, 8890.29it/s]    Evaluating paths:  34%|███▍      | 3581/10416 [00:00<00:00, 8902.22it/s]    Evaluating paths:  43%|████▎     | 4477/10416 [00:00<00:00, 8922.03it/s]    Evaluating paths:  52%|█████▏    | 5370/10416 [00:00<00:00, 8895.76it/s]    Evaluating paths:  60%|██████    | 6262/10416 [00:00<00:00, 8901.13it/s]    Evaluating paths:  69%|██████▉   | 7166/10416 [00:00<00:00, 8942.25it/s]    Evaluating paths:  77%|███████▋  | 8061/10416 [00:00<00:00, 8869.54it/s]    Evaluating paths:  86%|████████▌ | 8963/10416 [00:01<00:00, 8914.40it/s]    Evaluating paths:  95%|█████████▍| 9855/10416 [00:01<00:00, 8909.52it/s]    Evaluating paths: 100%|██████████| 10416/10416 [00:01<00:00, 8910.81it/s]




.. GENERATED FROM PYTHON SOURCE LINES 341-343

--------------


.. GENERATED FROM PYTHON SOURCE LINES 346-349

2. Non-linear Travel Time Tomography
------------------------------------


.. GENERATED FROM PYTHON SOURCE LINES 352-358

Now we demonstrate CoFI on a nonlinear iterative tomographic problem in
a cross borehole setting.

We use a different tomographic example from espresso. Here we import the
example module and plot the reference seismic model.


.. GENERATED FROM PYTHON SOURCE LINES 358-363

.. code-block:: Python


    nonlinear_tomo_example = espresso.FmmTomography()

    nonlinear_tomo_example.plot_model(nonlinear_tomo_example.good_model, with_paths=True,lw=0.5);




.. image-sg:: /tutorials/generated/images/sphx_glr_travel_time_tomography_005.png
   :alt: travel time tomography
   :srcset: /tutorials/generated/images/sphx_glr_travel_time_tomography_005.png
   :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 .. code-block:: none

     New data set has:
     10  receivers
     10  sources
     100  travel times
     Range of travel times:  0.008911182496368759 0.0153757024856463 
     Mean travel time: 0.01085811731230709

    <Axes: xlabel='x (km)', ylabel='y (km)'>



.. GENERATED FROM PYTHON SOURCE LINES 368-371

Solving the tomographic system with optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. GENERATED FROM PYTHON SOURCE LINES 374-432

Now we solve the tomographic system of equations using either CoFI’s
optimization method interface, or its iterative matrix-solver interface.

**For the optimization interface:**

We choose an objective function of the form.

.. math:: \phi(\mathbf{d},\mathbf{s}) = \frac{1}{\sigma^2}|| \mathbf{d} - \mathbf{g}(\mathbf{s})||_2^2 + \lambda_1 ||\mathbf{s}- \mathbf{s}_{0}||_2^2  + \lambda_2 ||D~\mathbf{s}||_2^2

where :math:`\mathbf{g}(\mathbf{s})` represents the predicted travel
times in the slowness model :math:`\mathbf{s}`, :math:`\sigma^2` is the
noise variance on the travel times, :math:`(\lambda_1,\lambda_2)` are
weights of damping and smoothing regularization terms respectively,
:math:`\mathbf{s}_{0}` is the reference slowness model provided by the
espresso example, and :math:`D` is a second derivative finite difference
stencil for the slowness model with shape ``model_shape``.

In the set up below this objective function is defined outside of CoFI
in the function ``objective_func`` together with its gradient and
Hessian, ``gradient`` and ``hessian`` with respect to slowness
parameters. For convenience the regularization terms are constructed
with CoFI utility routine ``QuadraticReg``.

For the optimization case CoFI passes ``objective_func`` and optionally
the ``gradient`` and ``Hessian`` functions to a thrid party optimization
backend tool such as ``scipy.minimize`` to produce a solution.

**For the iterative matrix solver interface:**

For convenience, CoFI also has its own Gauss-Newton Solver for
optimization of a general objective function of the form.

.. math::


   \phi(\mathbf{d},\mathbf{s}) = \psi((\mathbf{d},\mathbf{s}) + \sum_{r=1}^R \lambda_r \chi_r(\mathbf{s}),

where :math:`\psi` represents a data misfit term, and :math:`\chi_r` one
or more regularization terms, with weights :math:`\lambda_r`. The
objective function above is a special case of this. In general an
iterative Gauss-Newton solver takes the form

.. math::


   \mathbf{s}_{k+1} = \mathbf{s}_{k} - \cal{H}^{-1}(\mathbf{s}_k) \nabla \phi(\mathbf{s}_k), \quad {(k=0,1,\dots)},

where :math:`\cal{H}(\mathbf{s}_k)` is the Hessian of the objective
function, and :math:`\nabla \phi(\mathbf{s}_k)` its gradient evaluated
at the model :math:`\mathbf{s}_k`.

For the objective function above this becomes the simple iterative
matrix solver

.. math::  \mathbf{s}_{k+1} = \mathbf{s}_k + (A^T C_d^{-1}A + \lambda_2\mathbf{I} +\lambda_2D^TD )^{-1} [A^T C_d^{-1} (\mathbf{d} - g(\mathbf{s}_k)) -  \lambda_2 (\mathbf{s - s}_{0}) - \lambda_2 D^TD \mathbf{s}], \quad (k=0,1,\dots)

with :math:`C_d^{-1} = \sigma^{-2} I`.


.. GENERATED FROM PYTHON SOURCE LINES 435-438

Step 1. Define CoFI ``BaseProblem``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. GENERATED FROM PYTHON SOURCE LINES 438-445

.. code-block:: Python


    # get problem information from  espresso FmmTomography
    model_size = nonlinear_tomo_example.model_size               # number of model parameters
    model_shape = nonlinear_tomo_example.model_shape             # 2D spatial grid shape
    data_size = nonlinear_tomo_example.data_size                 # number of data points
    ref_start_slowness = nonlinear_tomo_example.starting_model   # use the starting guess supplied by the espresso example








.. GENERATED FROM PYTHON SOURCE LINES 450-453

Here we define the baseproblem object and a starting velocity model
guess.


.. GENERATED FROM PYTHON SOURCE LINES 453-458

.. code-block:: Python


    # define CoFI BaseProblem
    nonlinear_problem = cofi.BaseProblem()
    nonlinear_problem.set_initial_model(ref_start_slowness)








.. GENERATED FROM PYTHON SOURCE LINES 463-465

Here we define regularization of the tomographic system.


.. GENERATED FROM PYTHON SOURCE LINES 465-480

.. code-block:: Python


    # add regularization: damping / flattening / smoothing
    damping_factor = 50
    smoothing_factor = 5e3
    reg_damping = damping_factor * cofi.utils.QuadraticReg(
        model_shape=model_shape, 
        weighting_matrix="damping", 
        reference_model=ref_start_slowness
    )
    reg_smoothing = smoothing_factor * cofi.utils.QuadraticReg(
        model_shape=model_shape,
        weighting_matrix="smoothing"
    )
    reg = reg_damping + reg_smoothing








.. GENERATED FROM PYTHON SOURCE LINES 482-510

.. code-block:: Python


    def objective_func(slowness, reg, sigma, data_subset=None):
        if data_subset is None: 
            data_subset = np.arange(0, nonlinear_tomo_example.data_size)
        ttimes = nonlinear_tomo_example.forward(slowness)
        residual = nonlinear_tomo_example.data[data_subset] - ttimes[data_subset]
        data_misfit = residual.T @ residual / sigma**2
        model_reg = reg(slowness)
        return  data_misfit + model_reg

    def gradient(slowness, reg, sigma, data_subset=None):
        if data_subset is None: 
            data_subset = np.arange(0, nonlinear_tomo_example.data_size)
        ttimes, A = nonlinear_tomo_example.forward(slowness, return_jacobian=True)
        ttimes = ttimes[data_subset]
        A = A[data_subset]
        data_misfit_grad = -2 * A.T @ (nonlinear_tomo_example.data[data_subset] - ttimes) / sigma**2
        model_reg_grad = reg.gradient(slowness)
        return  data_misfit_grad + model_reg_grad

    def hessian(slowness, reg, sigma, data_subset=None):
        if data_subset is None: 
            data_subset = np.arange(0, nonlinear_tomo_example.data_size)
        A = nonlinear_tomo_example.jacobian(slowness)[data_subset]
        data_misfit_hess = 2 * A.T @ A / sigma**2 
        model_reg_hess = reg.hessian(slowness)
        return data_misfit_hess + model_reg_hess








.. GENERATED FROM PYTHON SOURCE LINES 512-519

.. code-block:: Python


    sigma = 0.00001                   # Noise is 1.0E-4 is ~5% of standard deviation of initial travel time residuals

    nonlinear_problem.set_objective(objective_func, args=[reg, sigma, None])
    nonlinear_problem.set_gradient(gradient, args=[reg, sigma, None])
    nonlinear_problem.set_hessian(hessian, args=[reg, sigma, None])








.. GENERATED FROM PYTHON SOURCE LINES 524-527

Step 2. Define CoFI ``InversionOptions``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. GENERATED FROM PYTHON SOURCE LINES 527-538

.. code-block:: Python


    nonlinear_options = cofi.InversionOptions()

    # cofi's own simple newton's matrix-based optimization solver
    nonlinear_options.set_tool("cofi.simple_newton")
    nonlinear_options.set_params(num_iterations=5, step_length=1, verbose=True)

    # scipy's Newton-CG solver (alternative approach with similar results)
    # nonlinear_options.set_tool("scipy.optimize.minimize")
    # nonlinear_options.set_params(method="Newton-CG", options={"xtol":1e-16})








.. GENERATED FROM PYTHON SOURCE LINES 540-543

.. code-block:: Python


    nonlinear_options.summary()





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    =============================
    Summary for inversion options
    =============================
    Solving method: None set
    Use `suggest_solving_methods()` to check available solving methods.
    -----------------------------
    Backend tool: `<class 'cofi.tools._cofi_simple_newton.CoFISimpleNewton'>` - CoFI's own solver - simple Newton's approach (for testing mainly)
    References: ['https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization']
    Use `suggest_tools()` to check available backend tools.
    -----------------------------
    Solver-specific parameters: 
    num_iterations = 5
    step_length = 1
    verbose = True
    Use `suggest_solver_params()` to check required/optional solver-specific parameters.




.. GENERATED FROM PYTHON SOURCE LINES 548-551

Step 3. Define CoFI ``Inversion`` and run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. GENERATED FROM PYTHON SOURCE LINES 551-556

.. code-block:: Python


    nonlinear_inv = cofi.Inversion(nonlinear_problem, nonlinear_options)
    nonlinear_inv_result = nonlinear_inv.run()
    nonlinear_tomo_example.plot_model(nonlinear_inv_result.model);




.. image-sg:: /tutorials/generated/images/sphx_glr_travel_time_tomography_006.png
   :alt: travel time tomography
   :srcset: /tutorials/generated/images/sphx_glr_travel_time_tomography_006.png
   :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    Iteration #0, updated objective function value: 1787.077540309464
    Iteration #1, updated objective function value: 121.06987606708292
    Iteration #2, updated objective function value: 5.825780480486444
    Iteration #3, updated objective function value: 3.671788666778372
    Iteration #4, updated objective function value: 1.607554713000219

    <Axes: xlabel='x (km)', ylabel='y (km)'>



.. GENERATED FROM PYTHON SOURCE LINES 561-563

Now lets plot the true model for comparison.


.. GENERATED FROM PYTHON SOURCE LINES 563-566

.. code-block:: Python


    nonlinear_tomo_example.plot_model(nonlinear_tomo_example.good_model);




.. image-sg:: /tutorials/generated/images/sphx_glr_travel_time_tomography_007.png
   :alt: travel time tomography
   :srcset: /tutorials/generated/images/sphx_glr_travel_time_tomography_007.png
   :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 .. code-block:: none


    <Axes: xlabel='x (km)', ylabel='y (km)'>



.. GENERATED FROM PYTHON SOURCE LINES 571-615

Challenge: Change the number of tomographic data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First try and repeat this tomographic reconstruction with fewer data and
see what the model looks like.

There are 100 raypaths in the full dataset and you can tell CoFI to
select a subset by passing an additional array of indices to the
functions that calculate objective, gradient and hessian.

|Upload to Jamboard 2|

Start from the code template below:

::

   # Set a subset of raypaths here
   data_subset = np.arange(<CHANGE ME>)

   # select BaseProblem
   my_own_nonlinear_problem = cofi.BaseProblem()
   my_own_nonlinear_problem.set_objective(objective_func, args=[reg, sigma, data_subset])
   my_own_nonlinear_problem.set_gradient(gradient, args=[reg, sigma, data_subset])
   my_own_nonlinear_problem.set_hessian(hessian, args=[reg, sigma, data_subset])
   my_own_nonlinear_problem.set_initial_model(ref_start_slowness)

   # run inversion with same options as previously
   my_own_inversion = cofi.Inversion(my_own_nonlinear_problem, nonlinear_options)
   my_own_result = my_own_inversion.run()

   # check results
   my_own_result.summary()

   # plot inverted model
   fig, paths = nonlinear_tomo_example.plot_model(my_own_result.model, return_paths=True)
   print(f"Number of paths used: {len(data_subset)}")

   # plot paths used
   for p in np.array(paths, dtype=object)[data_subset]:
       fig.axes[0].plot(p[:,0], p[:,1], "g", alpha=0.5,lw=0.5)

.. |Upload to Jamboard 2| image:: https://img.shields.io/badge/Click%20&%20upload%20your%20results%20to-Jamboard-lightgrey?logo=jamboard&style=for-the-badge&color=fcbf49&labelColor=edede9
   :target: https://jamboard.google.com/d/1TlHvC6_vHLDaZzWT3cG2hV3KCrh3M6aoxDVAJ2RGJBw/edit?usp=sharing


.. GENERATED FROM PYTHON SOURCE LINES 615-620

.. code-block:: Python


    # Copy the template above, Replace <CHANGE ME> with your answer










.. GENERATED FROM PYTHON SOURCE LINES 622-650

.. code-block:: Python


    #@title Solution

    # Set a subset of raypaths here
    data_subset = np.arange(30, 60)

    # select BaseProblem
    my_own_nonlinear_problem = cofi.BaseProblem()
    my_own_nonlinear_problem.set_objective(objective_func, args=[reg, sigma, data_subset])
    my_own_nonlinear_problem.set_gradient(gradient, args=[reg, sigma, data_subset])
    my_own_nonlinear_problem.set_hessian(hessian, args=[reg, sigma, data_subset])
    my_own_nonlinear_problem.set_initial_model(ref_start_slowness)

    # run inversion with same options as previously
    my_own_inversion = cofi.Inversion(my_own_nonlinear_problem, nonlinear_options)
    my_own_result = my_own_inversion.run()

    # check results
    my_own_result.summary()

    # plot inverted model
    fig, paths = nonlinear_tomo_example.plot_model(my_own_result.model, return_paths=True)
    print(f"Number of paths used: {len(data_subset)}")

    # plot paths used
    for p in np.array(paths, dtype=object)[data_subset]:
        fig.axes.plot(p[:,0], p[:,1], "g", alpha=0.5,lw=0.5)




.. image-sg:: /tutorials/generated/images/sphx_glr_travel_time_tomography_008.png
   :alt: travel time tomography
   :srcset: /tutorials/generated/images/sphx_glr_travel_time_tomography_008.png
   :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    Iteration #0, updated objective function value: 133.40921131094794
    Iteration #1, updated objective function value: 2.5732294086703247
    Iteration #2, updated objective function value: 0.4155731243675164
    Iteration #3, updated objective function value: 0.0044721955346204685
    Iteration #4, updated objective function value: 0.0004912544688082658
    Change in model parameters below tolerance, stopping.
    ============================
    Summary for inversion result
    ============================
    SUCCESS
    ----------------------------
    model: [0.00050057 0.00050052 0.00050046 ... 0.00051289 0.00051088 0.00050873]
    num_iterations: 4
    objective_val: 0.0004912544688082658
    n_obj_evaluations: 6
    n_grad_evaluations: 5
    n_hess_evaluations: 5
    Number of paths used: 30




.. GENERATED FROM PYTHON SOURCE LINES 655-721

Challenge: Change regularization settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the solution above we used ``damping_factor = 50``, and
``smoothing_factor = 5.0E-3`` and ``flattening_factor = 0``.

Experiment with these choices, e.g increasing all of them to say 100 and
repeat the tomographic solution to see how the model changes.

Try to turn off smoothing all together but retain damping and flattening
and see what happens.

With some choices you can force an under-determined problem which is not
solvable.

(Note that here we revert back to using all of the data by removing the
``data_subset`` argument to the objective function.)

To repeat this solver with other settings for smoothing and damping
strength. See the documentation for
`cofi.utils.QuadraticReg <https://cofi.readthedocs.io/en/latest/api/generated/cofi.utils.QuadraticReg.html>`__.

|Upload to Jamboard 3|

You can start from the template below:

::

   # change the combination of damping, flattening and smoothing regularizations
   damping_factor = <CHANGE ME>                # select damping factor here to force solution toward reference slowness model 
   flattening_factor = <CHANGE ME>             # increase flattening factor here to force small first derivatives in slowness solution
   smoothing_factor = <CHANGE ME>              # increase smoothing factor here to force small second derivatives in slowness solution

   reg_damping = damping_factor * cofi.utils.QuadraticReg(
       model_shape=model_shape,
       weighting_matrix="damping",
       reference_model=ref_start_slowness
   )
   reg_flattening = flattening_factor * cofi.utils.QuadraticReg(
       model_shape=model_shape,
       weighting_matrix="flattening"
   )
   reg_smoothing = smoothing_factor * cofi.utils.QuadraticReg(
       model_shape=model_shape,
       weighting_matrix="smoothing"
   )
   my_own_reg = reg_damping + reg_flattening + reg_smoothing

   # set Baseproblem
   my_own_nonlinear_problem = cofi.BaseProblem()
   my_own_nonlinear_problem.set_objective(objective_func, args=[my_own_reg, sigma, None])
   my_own_nonlinear_problem.set_gradient(gradient, args=[my_own_reg, sigma, None])
   my_own_nonlinear_problem.set_hessian(hessian, args=[my_own_reg, sigma, None])
   my_own_nonlinear_problem.set_initial_model(ref_start_slowness.copy())

   # run inversion with same options as previously
   my_own_inversion = cofi.Inversion(my_own_nonlinear_problem, nonlinear_options)
   my_own_result = my_own_inversion.run()

   # check results
   fig = nonlinear_tomo_example.plot_model(my_own_result.model)
   fig.suptitle(f"Damping {damping_factor}, Flattening {flattening_factor}, Smoothing {smoothing_factor}");

.. |Upload to Jamboard 3| image:: https://img.shields.io/badge/Click%20&%20upload%20your%20results%20to-Jamboard-lightgrey?logo=jamboard&style=for-the-badge&color=fcbf49&labelColor=edede9
   :target: https://jamboard.google.com/d/15FrdSczK_TK_COOLxfSJZ5CWMzH3qMoQKySJTAp5n-4/edit?usp=sharing


.. GENERATED FROM PYTHON SOURCE LINES 721-726

.. code-block:: Python


    # Copy the template above, Replace <CHANGE ME> with your answer










.. GENERATED FROM PYTHON SOURCE LINES 728-766

.. code-block:: Python


    #@title Reference Solution

    # change the combination of damping, flattening and smoothing regularizations
    damping_factor = 100                # select damping factor here to force solution toward reference slowness model 
    flattening_factor = 100             # increase flattening factor here to force small first derivatives in slowness solution
    smoothing_factor = 0                # increase smoothing factor here to force small second derivatives in slowness solution

    reg_damping = damping_factor * cofi.utils.QuadraticReg(
        model_shape=model_shape,
        weighting_matrix="damping",
        reference_model=ref_start_slowness
    )
    reg_flattening = flattening_factor * cofi.utils.QuadraticReg(
        model_shape=model_shape,
        weighting_matrix="flattening"
    )
    reg_smoothing = smoothing_factor * cofi.utils.QuadraticReg(
        model_shape=model_shape,
        weighting_matrix="smoothing"
    )
    my_own_reg = reg_damping + reg_flattening + reg_smoothing

    # set Baseproblem
    my_own_nonlinear_problem = cofi.BaseProblem()
    my_own_nonlinear_problem.set_objective(objective_func, args=[my_own_reg, sigma, None])
    my_own_nonlinear_problem.set_gradient(gradient, args=[my_own_reg, sigma, None])
    my_own_nonlinear_problem.set_hessian(hessian, args=[my_own_reg, sigma, None])
    my_own_nonlinear_problem.set_initial_model(ref_start_slowness.copy())

    # run inversion with same options as previously
    my_own_inversion = cofi.Inversion(my_own_nonlinear_problem, nonlinear_options)
    my_own_result = my_own_inversion.run()

    # check results
    ax = nonlinear_tomo_example.plot_model(my_own_result.model)
    ax.get_figure().suptitle(f"Damping {damping_factor}, Flattening {flattening_factor}, Smoothing {smoothing_factor}");




.. image-sg:: /tutorials/generated/images/sphx_glr_travel_time_tomography_009.png
   :alt: Damping 100, Flattening 100, Smoothing 0
   :srcset: /tutorials/generated/images/sphx_glr_travel_time_tomography_009.png
   :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    Iteration #0, updated objective function value: 1697.7749129381455
    Iteration #1, updated objective function value: 1645.6268255379405
    Iteration #2, updated objective function value: 710.1208786960945
    Iteration #3, updated objective function value: 715.5022745101638
    Iteration #4, updated objective function value: 685.5111622410774

    Text(0.5, 0.98, 'Damping 100, Flattening 100, Smoothing 0')



.. GENERATED FROM PYTHON SOURCE LINES 771-776

--------------

Watermark
---------


.. GENERATED FROM PYTHON SOURCE LINES 776-782

.. code-block:: Python


    watermark_list = ["cofi", "espresso", "numpy", "scipy", "matplotlib"]
    for pkg in watermark_list:
        pkg_var = __import__(pkg)
        print(pkg, getattr(pkg_var, "__version__"))





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    cofi 0.2.7
    espresso 0.3.13
    numpy 1.24.4
    scipy 1.12.0
    matplotlib 3.8.3




.. GENERATED FROM PYTHON SOURCE LINES 783-783

sphinx_gallery_thumbnail_number = -1


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 32.668 seconds)


.. _sphx_glr_download_tutorials_generated_travel_time_tomography.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: travel_time_tomography.ipynb <travel_time_tomography.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: travel_time_tomography.py <travel_time_tomography.py>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
