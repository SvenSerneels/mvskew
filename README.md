# `mvskew`: Multivariate skew distribution samplers

This package provides access a set of multivariate distribution samplers. Implemented 
multivariate distributions are:
- normal 
- skew normal
- extended skew normal 
- Cauchy 
- skew Cauchy
- extended skew Cauchy 
- Student T 
- skew Student T
- extended skew T

Installation 
------------
Tha package is distributed through PyPI: 

    pip install mvskew 
    
    
Usage 
-----
While the package contains functions to call samplers directly and a broad set
of helper functions, the main feature of the package is the class object 
`multivariate_samplers`, which wraps around all core functionalities provided.
An example to use it: 

    # Set params 
    xi = np.ones(10)
    Omega = np.diag(np.ones(10))
    alpha = np.array([3,2,1,4,5,6,8,5,2,0])
    tau = -1
    #Sample from a multivariate skew T5 truncated about -1
    mvs = multivariate_samplers(distribution='T',disttype='skew', df=5)
    mvs.sample(20,xi,Omega,alpha,tau)
    
References
----------

The skew and extended skew families are described in detail in: 
[1] Azzalini, A., in collaboration with Capitanio, A., The Skew-Normal and Related Families, IMS Monograph series, Cambridge University Press, Cambridge, UK, 2014.
[2] Azzalini, A. and Dalla Valle, A. The Multivariate Skew-Normal Distribution, Biometrika, Vol. 83, No. 4 (1996), pp. 715-726. 

