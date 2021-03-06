# Langevin equations for landmark image registration with uncertainty

This Python code accompanies the paper [Langevin equations for landmark image registration with
uncertainty](http://arxiv.org/abs/1605.09276), S. Marsland and T. Shardlow (2016).

    Registration of images parameterised by landmarks provides a useful method of 
    describing shape variations by computing the minimum-energy time-dependent deformation
    field that flows one landmark set to the other. This is sometimes known as the geodesic
    interpolating spline and can be solved via a Hamiltonian boundary-value problem to give
    a diffeomorphic registration between images. However, small changes in the positions of
    the landmarks can produce large changes in the resulting diffeomorphism. We formulate
    a Langevin equation for looking at small random perturbations of this registration. The
    Langevin equation and three computationally convenient approximations are introduced
    and used as prior distributions. A Bayesian framework is then used to compute a posterior
    distribution for the registration, and also to formulate an average of multiple sets of
    landmarks.

 Example sets and further explanation about the code are found in the [supplement](https://github.com/tonyshardlow/reg_sde/blob/master/sde_imag_supplement.pdf).
