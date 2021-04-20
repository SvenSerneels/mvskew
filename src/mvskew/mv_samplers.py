#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:29:38 2020

@author: Sven Serneels, Ponalytics
"""

import sys
# import torch 
import inspect
import numpy as np
import warnings
import scipy.stats as spp
from collections import defaultdict
from ._mv_samplers_utils import _check_format
from ._mv_samplers_utils import *

class multivariate_samplers:
    
    """
    multivariate_samplers 
    ---------------------
    
    Class to access a set of multivariate distribution samplers. Implemented 
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
    
    Each of these distributions can be sampled through this class object by 
    setting the appropriate parameters. 
    
    Parameters: 
        
        distribution, str, 'normal','Cauchy','Student' or 'T'
        disttype, str, 'symmetric' or 'skew'
        df, int, degrees of freedom (irrelevant for normal, resets to 1 for Cauchy)
        
    Methods: 
        
        sample(size, *args,**kwargs). Produces a sample, with input arguments:
            size, int, number of cases 
            xi, (d), (d,1) or (1,d) vector, matrix or tensor, location
            Omega, (d,d), matrix or tensor, scale 
            alpha, (d), (d,1) or (1,d) vector, matrix or tensor, slant 
            tau, int, float, or (d), (d,1) or (1,d) vector or matrix, 'truncation'
                tau parameter for the ESN. Setting tau=0 yields the pure skew 
                distribution, (Eq. 2.1 in [1]) , while any other value yields 
                an extended skew distribution (Eq. 2.39 in [1]). One value applies 
                to all variables, a vector sets it on a per-variable basis.
                
        get_params: get present set of parameters 
        set_params: set parameters of class object to new value
        
    Ancillary functions: 
        
        dp2cpMV: convert a set of direct parameters xi, Omega, alpha, tau into 
            centred parameters
        multivariate_t_density 
        sampling functions for each combination accessible through the class 
        
        
        
    Example: 
        
        # Set params 
        xi = np.ones(10)
        Omega = np.diag(np.ones(10))
        alpha = np.array([3,2,1,4,5,6,8,5,2,0])
        tau = -1
        #Sample from a multivariate skew T5 truncated about -1
        mvs = multivariate_samplers(distribution='T',disttype='skew', df=5)
        mvs.sample(20,xi,Omega,alpha,tau)
        #Convert parameters to centred parameters: 
        cp = dp2cpMV(xi,Omega,alpha,tau,nu=5,family='ST',cp_type='auto',aux=True)
    
    Backend: 
        
        It is the intention to have the class fully functional for both numpy 
        and torch backends. However, presently the torch backend is only supported
        for the multivariate normal, Cauchy and Student T families, not for the 
        skew and extended skew variants. To use the torch backend, uncomment the 
        respective imports. 
        
    References
    
        The skew and extended skew families are described in detail in: 
        [1] Azzalini, A., in collaboration with Capitanio, A., 
            The Skew-Normal and Related Families, 
            IMS Monograph series,
            Cambridge University Press, 
            Cambridge, UK, 2014.
        [2] Azzalini, A. and Dalla Valle, A.  
            The Multivariate Skew-Normal Distribution
            Biometrika
            Vol. 83, No. 4 (Dec., 1996), pp. 715-726. 
    
    """
    
    def __init__(self,distribution='normal',disttype='symmetric',df=1):
        self.distribution = distribution
        self.disttype = disttype
        self.df = df
        if self.distribution not in ['normal','Cauchy','Student','T']:
            raise(NotImplementedError("Only normal, Cauchy and T samplers available"))
        if self.disttype not in ['symmetric','skew']:
            raise(NotImplementedError("Only symmetric and skew samplers available"))
        self._dist = distribution
        if self.distribution == 'Cauchy':
            self._dist = 'T'
            self.df = 1
        if self.distribution == 'Student':
            self._dist = 'T'
        if self._dist == 'normal':
            if self.disttype == 'symmetric':
                self._function = multivariate_normal_sampler
            elif self.disttype == 'skew':
                self._function = multivariate_skewnormal_sampler
            else:
                self._function = None 
        elif self._dist == 'T':
            if self.disttype == 'symmetric':
                self._function = multivariate_T_sampler
            elif self.disttype == 'skew':
                self._function = multivariate_skew_T_sampler
                
    def sample(self,size,*args,**kwargs):
        
        if len(args) < 2: 
            raise(ValueError("Please provide appropriate distribution parameters"))
        if ((self.disttype == 'skew') and (len(args)<4)):
            raise(ValueError("Please provide skewness vector and tau parameter(s) for skew sampling"))
        if self._dist == 'normal':
            return(self._function(size,*args,**kwargs))
        elif self._dist == 'T':
            return(self._function(size,self.df,*args,**kwargs))
            
    @classmethod   
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=False):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        ------
        Copied from ScikitLlearn instead of imported to avoid 'deep=True'
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out
        
    def set_params(self, **params):
        """Set the parameters of this estimator.
        Copied from ScikitLearn, adapted to avoid calling 'deep=True'
        Returns
        -------
        self
        ------
        Copied from ScikitLlearn instead of imported to avoid 'deep=True'
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params()

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

def multivariate_normal_sampler(size,mu,Sigma,*args):
    
    """
    Sample from a multivariate normal distribution
    Inputs:
        size, int, number of cases to sample
        df, int, number of degrees of freedom (1 = multivariate Cauchy distribution)
        mu: (n), (n,1) or (1,n) vector, matrix or tensor, location 
        Sigma: (n,n) matrix or tensor, Scale matrix
    Outputs: 
        Z, (size,n), matrix or tensor, the sample. 
        
    Works both with a numpy and pytorch backend
    
    This function is just a wrapper for the corresponding numpy and torch functions
    
    """
    
    # *args is only there to allow making a call with alpha and tau that have no effect
    
    tmu = type(mu)
    if len(mu.shape) > 1:
        n,p = mu.shape
        if (n>1) and (p>1): 
            raise(ValueError("Please provide (n,1), (1,n) or (n,) shaped data."))
        else: 
            if p>n: 
                n = p
                mu = mu.reshape(p)
            else:
                mu = mu.reshape(n)
    if 'torch' in sys.modules:
        if tmu == torch.Tensor: 
            smnv = torch.distributions.MultivariateNormal(loc=mu,
                                                      covariance_matrix=Sigma)
            return(smnv.sample((size,)))
    elif tmu == np.ndarray:
        return(np.random.multivariate_normal(mu,Sigma,size=size))


def multivariate_T_sampler(size,df,mu,Sigma,*args,calcopt='SVD'):
    
    """
    Sample from a multivariate Student's t distribution
    Inputs:
        size, int, number of cases to sample
        df, int, number of degrees of freedom (1 = multivariate Cauchy distribution)
        mu: (n), (n,1) or (1,n) vector, matrix or tensor, location 
        Sigma: (n,n) matrix or tensor, Scale matrix
    Outputs: 
        T, (size,n), matrix or tensor, the sample. 
        
    Works both with a numpy and pytorch backend
    
    """
    
    # *args is only there to allow making a call with alpha and tau that have no effect
    
    tmu = type(mu)
    if tmu == np.matrix: 
        mu = np.array(mu)
    
    if len(mu.shape) > 1:
        n,p = mu.shape
        if (n>1) and (p>1): 
            raise(ValueError("Please provide (n,1), (1,n) or (n,) shaped data."))
        else: 
            if p>n: 
                n = p
                mu = mu.reshape(p)
            else:
                mu = mu.reshape(n)
    else:
        n = mu.shape[0]
    
    tsigma = type(Sigma)
    if tsigma != tmu: 
        Sigma = tmu(Sigma)
    
    m,p = Sigma.shape
    if p !=n: 
        raise(ValueError("Location and Covariance Matrix Dimensions mismatch"))
        
    if 'torch' in sys.modules:
        if tmu == torch.Tensor: 
            smnv = torch.distributions.MultivariateNormal(loc=torch.zeros(n),
                                                      covariance_matrix=torch.eye(n))
            chisq = torch.distributions.Chi2(df)
            Z = smnv.sample((size,))
            U,S,V = Sigma.svd()
            sqrt_Sigma = U.mm(torch.diag(torch.sqrt(S))).mm(V)
            T = mu + 1/(torch.sqrt(chisq.sample((size,))/df)).reshape((size,1)) * (Z.mm(sqrt_Sigma))
        
    elif tmu == np.ndarray:
        if calcopt == 'SVD':
            Z = np.random.multivariate_normal(np.zeros(n),
                                          np.diag(np.ones(n)),
                                          size=size)
            chisq = np.random.chisquare(df,size=size)
            U,S,V = np.linalg.svd(Sigma)
            sqrt_Sigma = np.matmul(np.matmul(U,np.diag(np.sqrt(S))),V)
            T = mu + 1/(np.sqrt(chisq/df)).reshape((size,1)) * np.matmul(Z,sqrt_Sigma)
        else: 
            Z = np.random.multivariate_normal(np.zeros(n),
                                          Sigma,
                                          size=size)
            chisq = np.random.chisquare(df,size=size)
            T = mu + 1/(np.sqrt(chisq/df)).reshape((size,1)) * Z
        
    else:
        raise(ValueError("Only numpy and torch backends supported"))
        
    return(T)
    
    
    
def multivariate_skewnormal_sampler(size,xi, Omega, alpha, tau):
    
    """
    Sample from a multivariate skew normal distribution based on the additive 
    approach 
    Inputs: size, int, number of cases to sample
            xi, (n,), (n,1) or (1,n) matrix or array, location
            Omega, (n,n), matrix, Scale
            alpha, (n,), (n,1) or (1,n) matrix or array, skewness
            tau, int, float, (1,), (1,1), (n,), (n,1) or (1,n), matrix or array, tau parameter(s)
    Output: the sample as a (size,n) matrix
    """
    
    if ((type(size) not in [int,np.int_]) or (size <= 0)):
        raise(ValueError("Size must be a strictly positive integer"))
    if type(tau) in [int,float,np.int_,np.float_]:
        tau = np.array([tau])
    
    xi = _check_format(xi)
    alpha = _check_format(alpha)
    tau = _check_format(tau)
    
    (beta, Sigma, gamma1, tau, omega, R, O_inv, Ocor, O_pcor, \
     lambhdha, Psi, delta, delta_star, alpha_star, gamma1M, gamma2M) = \
     msn_dp2cp(xi,Omega,alpha,tau,aux=True)
    d = alpha.shape[0]
    y = np.matmul(np.random.normal(size=(size,d)),np.linalg.cholesky(Psi)) # each row is N_d(0,Psi)
    if (tau==np.array([0])).all():    
        z = np.array(np.matmul(np.random.normal(size=(size,d)),np.linalg.cholesky(Psi)))
        where_sign_flip = (np.abs(z) > np.array(gamma1).reshape(-1)*np.abs(np.array(y)))
        z[np.where(where_sign_flip)[0],:] *= -1
    else: 
        start = spp.norm.cdf(-tau[0])
        truncN = spp.norm.ppf(spp.uniform.rvs(start,1-start,size=size))
        truncN = np.tile(truncN.reshape((size,1)),(1,d))
        z = np.multiply(np.multiply(delta, truncN) + np.sqrt(1-np.square(delta)),y)
    
    y = beta + np.multiply(omega,z)
    
    return(y)
    
    
def multivariate_skew_T_sampler(size, df, xi, Omega, alpha, tau):
    
    """
    Sample from a multivariate skew T distribution based on the additive 
    approach 
    Inputs: size, int, number of cases to sample
            df, int, number of degrees of freedom (1 = multivariate Cauchy distribution)
            xi, (n,), (n,1) or (1,n) matrix or array, location
            Omega, (n,n), matrix, Scale
            alpha, (n,), (n,1) or (1,n) matrix or array, skewness
            tau, int, float, (1,), (1,1), (n,), (n,1) or (1,n), matrix or array, tau parameter(s)
    Output: the sample as a (size,n) matrix
    """
    
    if ((type(df) not in [int,np.int_]) or (df <= 0)):
        raise(ValueError("Degrees of freedom must be a strictly positive integer"))
    if ((type(size) not in [int,np.int_]) or (size <= 0)):
        raise(ValueError("Size must be a strictly positive integer"))
    
    if type(tau) in [int,float,np.int_,np.float_]:
        tau = np.array([tau])
    
    xi = _check_format(xi)
    alpha = _check_format(alpha)
    tau = _check_format(tau)
    
    if (np.isinf(alpha).any()): 
        raise(ValueError("Inf's in alpha are not allowed"))
    d = alpha.shape[0]
    if (df == np.infty):
        x = 1 
    else: 
        x = (spp.chi2.rvs(df,size=size)/df).reshape((size,1))
    z = multivariate_skewnormal_sampler(size, np.repeat(0,d), Omega, alpha, tau)
    y = np.repeat(xi.reshape((1,d)),size,axis=0) + np.divide(z,np.repeat(np.sqrt(x),d,axis=1))
    
    return(y)
    
     
def dp2cpMV(xi,Omega,alpha,tau=0, nu=np.infty, family='SN', cp_type="proper", aux=False, upto=4):
    
    """
    Transformation of direct distribution parameters to centred distribution 
    parameters for all distribution families in scope. 
    Inputs: 
        xi, (n,), (n,1) or (1,n) matrix or array, location
        Omega, (n,n), matrix, Scale
        alpha, (n,), (n,1) or (1,n) matrix or array, skewness
        tau, (n,), (n,1) or (1,n) matrix or array, tau parameter(s)
        nu, int, degrees of freedom
        family, str, distribution family, options: 'SN' (skew normal), 'ST' (skew T)
            or 'SC' (skew Cauchy)
        cp_type, str, type of centred parameters, "proper" or "approx"; 
            "proper" can only be computged if nu > upto 
        aux, bool, to report auxiliary estimates
        upto, int, number of moments to compute in correction (default = max =4).
    Output
        a tuple comtaining: transformed location, scale, skewness and input skewness
        if aux=True, a longer tuple containing auxiliary estimates.

    """
    
    family = family.upper()
    
    if (cp_type not in ("auto","proper","pseudo")):
        raise(ValueError(f"CP type {cp_type} is not supported"))
    
    if (family not in ("SN","ST","SC")):
        raise(ValueError(f"family {family} is not supported"))
        
    if type(tau) in [int,float,np.int_,np.float_]:
        tau = np.array([tau])
        
    if (family is ("SN")):  
        if (cp_type == "pseudo"): 
            warnings.warn("'cp_type=pseudo' makes no sense for the skew normal family")
        cp = msn_dp2cp(xi,Omega,alpha,tau,aux=aux)
        
    if (family in ("SC","ST")):
        if(cp_type=="auto"): 
             if((family == "SC") or (nu <= 4)): 
                 cp_type = "pseudo" 
        else: 
                cp_type = "proper"
                
        if (family == "SC"): 
            nu = 1
        cp = mst_dp2cp(xi,Omega,alpha,tau,nu,upto,cp_type=cp_type,aux=aux)
    if (cp is None): 
        warnings.warn("no CP could be found")
    return(cp)
    
    
def op2dp(xi, Psi, lambhdha): 
    
    """
    Transformation from original distribution parameterization [2] to 
    actual direct distrbution parameters [1]
    for all distribution families in scope. 
    Inputs: 
        xi, (n,), (n,1) or (1,n) matrix or array, location
        Psi, (n,n), matrix, Shape (original parametrization)
        lambhdha, (n,), (n,1) or (1,n) matrix or array, marginal skewness
    Output
        a tuple comtaining: transformed location, scale, slant.

    """
    
    if type(xi) in [int,float,np.int_,np.float_]:
        
        delta = delta_etc(xi)
        Omega = np.divide(Psi,np.sqrt(1 - np.square(delta)))
        alpha = lambhdha
        
    else:
        
        psi = np.sqrt(np.diag(Psi))
        delta = np.divide(lambhdha,np.sqrt(1 + np.square(lambhdha)))
        D_delta = np.sqrt(1 - np.square(delta))
        Psi_bar = cov2cor(Psi)
        omega = np.divide(psi,D_delta)
        tmp = np.matmul(np.linalg.inv(Psi_bar),lambhdha)
        Omega = Psi + np.outer(np.multiply(psi,lambhdha), np.multiply(psi,lambhdha))  # four lines before (5.30)
        alpha = np.divide(np.divide(tmp,D_delta),np.sqrt(1 + np.sum(np.multiply(lambhdha,tmp))))  # (5.22)
        
    return(xi,Omega,alpha)
    
        
def multivariate_T_density(x,mu,Sigma,df,d):
    """
    Multivariate t-student density:
    output:
        the density of the given element
    input:
        x = parameter (d dimensional numpy array or scalar)
        mu = mean (d dimensional numpy array or scalar)
        Sigma = scale matrix (dxd numpy array)
        df = degrees of freedom
        d: dimension
    """
    Num = gamma(1. * (d+df)/2)
    Denom = ( gamma(1.*df/2) * pow(df*pi,1.*d/2) * pow(np.linalg.det(Sigma),1./2) * pow(1 + (1./df)*np.dot(np.dot((x - mu),np.linalg.inv(Sigma)), (x - mu)),1.* (d+df)/2))
    d = 1. * Num / Denom 
    return d