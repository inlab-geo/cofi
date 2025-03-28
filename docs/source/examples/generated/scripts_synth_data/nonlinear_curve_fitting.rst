
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "examples/generated/scripts_synth_data/nonlinear_curve_fitting.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_examples_generated_scripts_synth_data_nonlinear_curve_fitting.py>`
        to download the full example code

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_generated_scripts_synth_data_nonlinear_curve_fitting.py:


Non-linear Curve Fitting
========================

.. GENERATED FROM PYTHON SOURCE LINES 9-14

|Open In Colab|

.. |Open In Colab| image:: https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670
   :target: https://colab.research.google.com/github/inlab-geo/cofi-examples/blob/main/examples/nonlinear_curve_fitting/nonlinear_curve_fitting.ipynb


.. GENERATED FROM PYTHON SOURCE LINES 17-24

If you are running this notebook locally, make sure you’ve followed
`steps
here <https://github.com/inlab-geo/cofi-examples#run-the-examples-with-cofi-locally>`__
to set up the environment. (This
`environment.yml <https://github.com/inlab-geo/cofi-examples/blob/main/envs/environment.yml>`__
file specifies a list of packages required to run the notebooks)


.. GENERATED FROM PYTHON SOURCE LINES 27-36

.. raw:: html

   <!-- TODO - background introduction for this problem. -->

In this notebook, we use ``cofi`` to run a non-linear curve fitting
problem:

.. math:: f(x)=\exp(a*x)+b


.. GENERATED FROM PYTHON SOURCE LINES 39-42

Import modules
--------------


.. GENERATED FROM PYTHON SOURCE LINES 42-51

.. code-block:: Python


    # -------------------------------------------------------- #
    #                                                          #
    #     Uncomment below to set up environment on "colab"     #
    #                                                          #
    # -------------------------------------------------------- #

    # !pip install -U cofi








.. GENERATED FROM PYTHON SOURCE LINES 53-62

.. code-block:: Python


    import numpy as np
    import matplotlib.pyplot as plt
    import arviz as az

    from cofi import BaseProblem, InversionOptions, Inversion

    np.random.seed(42)








.. GENERATED FROM PYTHON SOURCE LINES 67-70

Define the problem
------------------


.. GENERATED FROM PYTHON SOURCE LINES 70-84

.. code-block:: Python


    def my_forward(m, x):
        return np.exp(m[0] * x) + m[1]

    def my_jacobian(m, x):
        G=np.zeros([len(x),2])
        G[:,0]=x*np.exp(m[0]*x) # derivative with respect to m[0] 
        G[:,1]=np.ones(len(x))  # derivtavie with respect to m[1]
        return G

    def my_residuals(m, x, y):
        yhat = my_forward(m,x)
        return yhat-y








.. GENERATED FROM PYTHON SOURCE LINES 86-95

.. code-block:: Python


    # Choose the "true" parameters.
    a_true = 5.0
    b_true = 4.0
    f_true = 0.1

    m_true = [a_true,b_true]
    mf_true= [a_true,b_true,f_true]








.. GENERATED FROM PYTHON SOURCE LINES 97-112

.. code-block:: Python


    # Generate some synthetic data from the model.
    N = 50
    x = np.sort(1 * np.random.rand(N))
    yerr = 0.1 + 0.5 * np.random.rand(N)
    y = my_forward(m_true,x)
    y += np.abs(f_true * y) * np.random.randn(N)
    y += yerr * np.random.randn(N)
    plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
    x0 = np.linspace(0, 1, 500)
    plt.plot(x0, my_forward(m_true,x0), "k", alpha=0.3, lw=3)
    plt.xlim(0, 1)
    plt.xlabel("x")
    plt.ylabel("y");




.. image-sg:: /examples/generated/scripts_synth_data/images/sphx_glr_nonlinear_curve_fitting_001.png
   :alt: nonlinear curve fitting
   :srcset: /examples/generated/scripts_synth_data/images/sphx_glr_nonlinear_curve_fitting_001.png
   :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 .. code-block:: none


    Text(38.097222222222214, 0.5, 'y')



.. GENERATED FROM PYTHON SOURCE LINES 114-124

.. code-block:: Python


    # define the problem in cofi
    inv_problem = BaseProblem()
    inv_problem.name = "Curve Fitting"
    inv_problem.set_data(y)
    inv_problem.set_forward(my_forward, args=[x])
    inv_problem.set_jacobian(my_jacobian, args=[x])
    inv_problem.set_residual(my_residuals, args=[x,y])
    inv_problem.set_initial_model([3,3])








.. GENERATED FROM PYTHON SOURCE LINES 129-132

Example 1. least squares optimizer (levenber marquardt)
-------------------------------------------------------


.. GENERATED FROM PYTHON SOURCE LINES 132-137

.. code-block:: Python


    inv_options = InversionOptions()
    inv_options.set_tool("scipy.optimize.least_squares")
    inv_options.set_params(method="lm", max_nfev=10)








.. GENERATED FROM PYTHON SOURCE LINES 139-148

.. code-block:: Python


    ######## Run it
    inv = Inversion(inv_problem, inv_options)
    inv_result = inv.run()

    ######## Check result
    print(f"The inversion result from `scipy.optimize.minimize`: {inv_result.model}\n")
    inv_result.summary()





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    The inversion result from `scipy.optimize.minimize`: [5.06442618 3.54842172]

    ============================
    Summary for inversion result
    ============================
    SUCCESS
    ----------------------------
    cost: 751.5703778228749
    fun: [ 8.46834974e-02 -1.77230955e-02 -5.52853293e-01  8.89806503e-01
      2.91152920e-01 -6.80792317e-01 -1.14702071e+00 -2.15801090e-01
      1.82952940e-01 -5.26482030e-01 -7.76017779e-01 -5.59530381e-01
     -4.95847931e-01 -4.13394792e-01 -5.36314270e-01 -1.56467760e+00
      4.20608348e-01 -1.91245184e-01 -7.95757076e-02  4.30437727e-01
     -1.36307871e-02 -3.20414157e-01 -3.61292253e-01 -1.97016377e-01
      1.47256652e+00  1.95462598e-01  6.42560479e-01  1.17710109e+00
      1.82720280e-01 -5.85651733e-01 -4.32433161e+00 -4.33451431e-01
      1.59207006e-02  4.24747095e-01  5.23801008e+00  2.40244378e-01
     -2.85673020e-01 -6.65912029e+00  1.06971709e+00 -1.41328842e-01
      1.44236334e+00  7.70525925e+00 -4.25388813e+00 -1.75601284e+00
     -1.98652707e+00  1.44619318e+01 -9.86284710e+00  2.35903628e+01
     -2.98371685e-02 -2.11903105e+01]
    jac: [[2.28462443e-02 1.00000000e+00]
     [4.09307227e-02 1.00000000e+00]
     [5.87699128e-02 1.00000000e+00]
     [7.79481395e-02 1.00000000e+00]
     [9.04348484e-02 1.00000000e+00]
     [1.60175374e-01 1.00000000e+00]
     [2.26419173e-01 1.00000000e+00]
     [2.82725638e-01 1.00000000e+00]
     [3.43725582e-01 1.00000000e+00]
     [3.43820726e-01 1.00000000e+00]
     [4.04431999e-01 1.00000000e+00]
     [4.56634655e-01 1.00000000e+00]
     [4.64300862e-01 1.00000000e+00]
     [4.71420526e-01 1.00000000e+00]
     [5.48900988e-01 1.00000000e+00]
     [6.22385905e-01 1.00000000e+00]
     [9.59632427e-01 1.00000000e+00]
     [1.27285650e+00 1.00000000e+00]
     [1.28279178e+00 1.00000000e+00]
     [1.42031878e+00 1.00000000e+00]
     [1.42473141e+00 1.00000000e+00]
     [1.51128333e+00 1.00000000e+00]
     [2.34264052e+00 1.00000000e+00]
     [2.49621211e+00 1.00000000e+00]
     [3.85009064e+00 1.00000000e+00]
     [4.08975791e+00 1.00000000e+00]
     [4.59341502e+00 1.00000000e+00]
     [6.07964753e+00 1.00000000e+00]
     [6.95336935e+00 1.00000000e+00]
     [7.24310829e+00 1.00000000e+00]
     [7.48401301e+00 1.00000000e+00]
     [8.71405798e+00 1.00000000e+00]
     [1.19018190e+01 1.00000000e+00]
     [1.24136629e+01 1.00000000e+00]
     [1.26206405e+01 1.00000000e+00]
     [1.31778419e+01 1.00000000e+00]
     [1.35640163e+01 1.00000000e+00]
     [1.89839544e+01 1.00000000e+00]
     [2.18847690e+01 1.00000000e+00]
     [2.55534573e+01 1.00000000e+00]
     [2.98190139e+01 1.00000000e+00]
     [4.18720384e+01 1.00000000e+00]
     [4.84904621e+01 1.00000000e+00]
     [5.63991081e+01 1.00000000e+00]
     [6.96176523e+01 1.00000000e+00]
     [9.09334829e+01 1.00000000e+00]
     [1.15942416e+02 1.00000000e+00]
     [1.17246757e+02 1.00000000e+00]
     [1.28432017e+02 1.00000000e+00]
     [1.31826241e+02 1.00000000e+00]]
    grad: [1.46155217e-04 9.56170254e-10]
    optimality: 0.0001461552166447607
    active_mask: [0 0]
    nfev: 7
    njev: 5
    status: 2
    message: `ftol` termination condition is satisfied.
    model: [5.06442618 3.54842172]




.. GENERATED FROM PYTHON SOURCE LINES 153-156

Example 2. emcee
----------------


.. GENERATED FROM PYTHON SOURCE LINES 156-165

.. code-block:: Python


    sigma = 10                                     # common noise standard deviation
    Cdinv = np.eye(len(y))/(sigma**2)      # inverse data covariance matrix

    def my_log_likelihood(m,x,y,Cdinv):
        yhat = my_forward(m,x)
        residual = y-yhat
        return -0.5 * residual @ (Cdinv @ residual).T








.. GENERATED FROM PYTHON SOURCE LINES 167-176

.. code-block:: Python


    m_min = [0,0]             # lower bound for uniform prior
    m_max = [10,10]          # upper bound for uniform prior

    def my_log_prior(m,m_min,m_max):    # uniform distribution
        for i in range(len(m)):
            if m[i] < m_min[i] or m[i] > m_max[i]: return -np.inf
        return 0.0 # model lies within bounds -> return log(1)








.. GENERATED FROM PYTHON SOURCE LINES 178-184

.. code-block:: Python


    nwalkers = 12
    ndim = 2
    nsteps = 500
    walkers_start = np.array([5.,4.]) + 1e-1 * np.random.randn(nwalkers, ndim)








.. GENERATED FROM PYTHON SOURCE LINES 186-191

.. code-block:: Python


    inv_problem.set_log_prior(my_log_prior,args=[m_min,m_max])
    inv_problem.set_log_likelihood(my_log_likelihood,args=[x,y,Cdinv])
    inv_problem.set_model_shape(ndim)








.. GENERATED FROM PYTHON SOURCE LINES 193-206

.. code-block:: Python


    inv_options = InversionOptions()
    inv_options.set_tool("emcee")
    inv_options.set_params(nwalkers=nwalkers, nsteps=nsteps, initial_state=walkers_start)

    ######## Run it
    inv = Inversion(inv_problem, inv_options)
    inv_result = inv.run()

    ######## Check result
    print(f"The inversion result from `emcee`:")
    inv_result.summary()





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    The inversion result from `emcee`:
    ============================
    Summary for inversion result
    ============================
    SUCCESS
    ----------------------------
    sampler: <emcee.ensemble.EnsembleSampler object>
    blob_names: ['log_likelihood', 'log_prior']




.. GENERATED FROM PYTHON SOURCE LINES 208-212

.. code-block:: Python


    sampler = inv_result.sampler
    az_idata = inv_result.to_arviz()








.. GENERATED FROM PYTHON SOURCE LINES 214-218

.. code-block:: Python


    labels = ["m0", "m1"]
    az.plot_trace(az_idata);




.. image-sg:: /examples/generated/scripts_synth_data/images/sphx_glr_nonlinear_curve_fitting_002.png
   :alt: var_0, var_0, var_1, var_1
   :srcset: /examples/generated/scripts_synth_data/images/sphx_glr_nonlinear_curve_fitting_002.png
   :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 .. code-block:: none


    array([[<Axes: title={'center': 'var_0'}>,
            <Axes: title={'center': 'var_0'}>],
           [<Axes: title={'center': 'var_1'}>,
            <Axes: title={'center': 'var_1'}>]], dtype=object)



.. GENERATED FROM PYTHON SOURCE LINES 220-229

.. code-block:: Python


    _, axes = plt.subplots(2, 2, figsize=(14,10))
    az.plot_pair(
        az_idata.sel(draw=slice(300,None)), 
        marginals=True, 
        reference_values=dict(zip([f"var_{i}" for i in range(2)], m_true   )),
        ax = axes
    );




.. image-sg:: /examples/generated/scripts_synth_data/images/sphx_glr_nonlinear_curve_fitting_003.png
   :alt: nonlinear curve fitting
   :srcset: /examples/generated/scripts_synth_data/images/sphx_glr_nonlinear_curve_fitting_003.png
   :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 .. code-block:: none


    array([[<Axes: >, <Axes: >],
           [<Axes: xlabel='var_0', ylabel='var_1'>, <Axes: >]], dtype=object)



.. GENERATED FROM PYTHON SOURCE LINES 231-250

.. code-block:: Python


    flat_samples = sampler.get_chain(discard=300, thin=30, flat=True)
    inds = np.random.randint(len(flat_samples), size=100) # get a random selection from posterior ensemble
    _x_plot = np.linspace(0,1.0)
    _y_plot =  my_forward(m_true,_x_plot)
    plt.figure(figsize=(12,8))
    sample = flat_samples[0]
    _y_synth =  my_forward(sample,_x_plot)
    plt.plot(_x_plot, _y_synth, color="seagreen", label="Posterior samples",alpha=0.1)
    for ind in inds:
        sample = flat_samples[ind]
        _y_synth =  my_forward(sample,_x_plot)
        plt.plot(_x_plot, _y_synth, color="seagreen", alpha=0.1)
    plt.plot(_x_plot, _y_plot, color="darkorange", label="true model")
    plt.scatter(x, y, color="lightcoral", label="observed data")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend();




.. image-sg:: /examples/generated/scripts_synth_data/images/sphx_glr_nonlinear_curve_fitting_004.png
   :alt: nonlinear curve fitting
   :srcset: /examples/generated/scripts_synth_data/images/sphx_glr_nonlinear_curve_fitting_004.png
   :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 .. code-block:: none


    <matplotlib.legend.Legend object at 0x7f8252221b70>



.. GENERATED FROM PYTHON SOURCE LINES 255-268

--------------

Watermark
---------

.. raw:: html

   <!-- Feel free to add more modules in the watermark_list below, if more packages are used -->

.. raw:: html

   <!-- Otherwise please leave the below code cell unchanged -->


.. GENERATED FROM PYTHON SOURCE LINES 268-274

.. code-block:: Python


    watermark_list = ["cofi", "numpy", "scipy", "matplotlib", "emcee", "arviz"]
    for pkg in watermark_list:
        pkg_var = __import__(pkg)
        print(pkg, getattr(pkg_var, "__version__"))





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    cofi 0.2.7
    numpy 1.24.4
    scipy 1.12.0
    matplotlib 3.8.3
    emcee 3.1.4
    arviz 0.17.0




.. GENERATED FROM PYTHON SOURCE LINES 275-275

sphinx_gallery_thumbnail_number = -1


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 0.879 seconds)


.. _sphx_glr_download_examples_generated_scripts_synth_data_nonlinear_curve_fitting.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: nonlinear_curve_fitting.ipynb <nonlinear_curve_fitting.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: nonlinear_curve_fitting.py <nonlinear_curve_fitting.py>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
