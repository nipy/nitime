=======================================
Jack-knifing a Multitaper SDF estimator
=======================================

Assume there is a parameter :math:`\theta` that parameterizes a distribution, and that the set of random variables :math:`\lbrace Y_1, Y_2, ..., Y_n \rbrace` are i.i.d. according to that distribution.

The basic jackknifed estimator :math:`\tilde{\theta}` of some parameter :math:`\theta` is found through forming *pseudovalues* :math:`\hat{\theta}_i` based on the original set of samples. With *n* samples, there are *n* pseudovalues based on *n* "leave-one-out" sample sets.

General JN definitions
----------------------

| **simple sample estimate**
| :math:`\hat{\theta} = \dfrac{1}{n}\sum_i Y_i`
| **leave-one-out measurement**
| :math:`\hat{\theta}_{-i} = \dfrac{1}{n-1}\sum_{k \neq i}Y_k`
| **pseudovalues**
| :math:`\hat{\theta}_i = n\hat{\theta} - (n-1)\hat{\theta}_{-i}`

Now the jackknifed esimator is computed as

:math:`\tilde{\theta} = \dfrac{1}{n}\sum_i \hat{\theta}_i = n\hat{\theta} - \dfrac{n-1}{n}\sum_i \hat{\theta}_{-i}`

This estimator is known (?) to be distributed about the true parameter \theta approximately as a Student's t distribution, with standard error defined as

:math:`s^{2} = \dfrac{n-1}{n}\sum_i \left(\hat{\theta}_i - \tilde{\theta}\right)^{2}`

General Multitaper definition
-----------------------------

The general multitaper spectral density function (sdf) estimator, using *n* orthonormal tapers, combines the *n* :math:`\lbrace \hat{S}_i^{mt} \rbrace` sdf estimators, and takes the form

:math:`\hat{S}^{mt}(f) = \dfrac{\sum_{k} w_k(f)^2S^{mt}_k(f)}{\sum_{k} |w_k(f)|^2} = \dfrac{\sum_{k} w_k(f)^2S^{mt}_k(f)}{\lVert \vec{w}(f) \rVert^2}`

For instance, using discrete prolate spheroidal sequences (DPSS) windows, the :math:`\rbrace w_i \lbrace` set, in their simplest form, are the eigenvalues of the spectral concentration system. 

A natural choice for a *leave-one-out* measurement is (leaving out the dependence on argument *f*)

:math:`\ln\hat{S}_{-i}^{mt} = \ln\dfrac{\sum_{k \neq i} w_k^2S^{mt}_k}{\lVert \vec{w}_{-i} \rVert^2} = \ln\sum_{k \neq i} w_k^2S^{mt}_k - \ln\lVert \vec{w}_{-i} \rVert^2`

where :math:`\vec{w}_{-i}` is the vector of weights with the *ith* element set to zero. The natural log has been taken so that the estimate is distributed below and above :math:`S(f)` more evenly.

Multitaper Pseudovalues
-----------------------

I'm not quite clear on the form of the pseudovalues for multitaper combinations. 

One Option
``````````

The simple option is to weight the different *leave-one-out* measurements equally, which leads to

:math:`\ln\hat{S}_{i}^{mt} = n\ln\hat{S}^{mt} - (n-1)\ln\hat{S}_{-i}^{mt}`

And of course the estimate of :math:`S(f)` is given by

:math:`\ln\tilde{S}^{mt} (f) = \dfrac{1}{n}\sum_i \ln\hat{S}_i^{mt}(f)`

Another Option
``````````````

Another approach seems obvious which weights the *leave-one-out* measurements according to the length of :math:`\vec{w}_{-i}`. It would look something like this

| let
| :math:`g = {\lVert \vec{w} \rVert^2}`
| :math:`g_i = {\lVert \vec{w}_{-i} \rVert^2}`

Then the pseudovalues are

:math:`\ln\hat{S}_i^{mt} = \left(\ln\hat{S}^{mt} + \ln g\right) - \left(\ln\hat{S}_{-i}^{mt} + \ln g_i\right)`

and the jackknifed estimator is

:math:`\ln\tilde{S}^{mt} = \sum_i \ln\hat{S}_i^{mt} - \ln g`

and I would wager, the standard error is estimated as

:math:`s^2 = \dfrac{1}{n}\sum_i \left(\ln\hat{S}_i^{mt} - \ln\tilde{S}^{mt}\right)^2`

Consensus in Literature (??)
````````````````````````````

From what I can tell from a couple of sources from Thompson [#f1]_, [#f2]_, this is the approach for estimating the variance.

