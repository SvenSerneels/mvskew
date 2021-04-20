#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 20:07:11 2020

@author: Sven Serneels, Ponalytics
"""

import numpy as np
import scipy.stats as spp
import scipy.special as sps
import warnings


def _check_format(x):
    
    if type(x) == np.matrix:
        x = np.array(x)
        
    if len(x.shape) > 1:
        x = x.reshape(-1)
        
    return(x)
    
    
def mst_dp2cp(xi,Omega,alpha,tau=0,nu=1,upto=4,cp_type="proper", 
              symmetr=False, aux=False):
    
    """
    Transformation of direct distribution parameters to centred distribution 
    parameters for the skew T family. 
    Inputs: 
        xi, (n,), (n,1) or (1,n) matrix or array, location
        Omega, (n,n), matrix, Scale
        alpha, (n,), (n,1) or (1,n) matrix or array, skewness
        tau, (n,), (n,1) or (1,n) matrix or array, tau parameter(s)
        nu, int, degrees of freedom
        upto, int, number of moments to compute in correction (default = max =4)
        cp_type, str, type of centred parameters, "proper" or "approx"; 
            "proper" can only be computged if nu > upto
        symmetr, bool, to enforce symmetry. Defaults to False. 
        aux, bool, to report auxiliary estimates. 
    Output
        a tuple comtaining: transformed location, scale, skewness and input skewness
    Remark
        Works for univariate parameters, but they have to be entered as array 
        or matrix, e.g. 
        msn_dp2cp(np.array([1]),np.array([2]),np.array([1]))
    """
    
    xi = _check_format(xi)
    alpha = _check_format(alpha)
    tau = _check_format(tau)
    
    if((round(upto) != upto) or (upto < 1)): 
        raise(ValueError("'upto' must be positive integer"))
    if((nu <= upto) and (cp_type =="proper")): 
        warnings.warn("Proper type of correction cannot be computed if df < number of moments")
        return(None)
    if(cp_type == "proper"):
        if(nu <= upto): 
            warnings.warn("Centred parameters are not defined at " + str(nu) + " degrees of freedom")
            return(None)
        a = np.zeros(upto) 
    else:
        a = np.arange(1,upto+1) 
         
    d = Omega.shape[0]
    if symmetr:
        alpha = np.repeat(0,d) 
    omega = np.sqrt(np.diag(Omega))
    delta,alpha_star,delta_star,Ocor = delta_etc(alpha, Omega)
    mu0 = np.multiply(np.multiply(bleat(nu+a[0]),delta),omega)
    mu_2 = np.multiply(np.multiply(bleat(nu+a[1]),delta),omega)
    beta = xi + mu0
    if (upto > 1):
        Sigma = np.multiply(Omega,np.divide((nu+a[1]),(nu+a[1]-2))) - np.outer(mu_2, mu_2)
    else: 
        Sigma = Omega
    
    if ((upto > 2) and not(symmetr)):
        gamma1 =  st_gamma1(delta, nu+a[2]) 
    else: 
        gamma1 = None
    
    if (upto > 3):
        gamma2 =  mst_mardia(np.square(delta_star), nu+a[3], d)[1]
    else: 
        gamma2 = None
        
    if aux:
        if nu <= 3:
            warnings.warn("Mardia parameters can only be computed for df>=4")
            cp = (beta, Sigma, gamma1, gamma2, nu)
        else:
            mardia = mst_mardia(np.square(delta_star), nu, d)
            cp = (beta, Sigma, gamma1, gamma2, nu, omega, Ocor, delta,
                delta_star, alpha_star, mardia)
    else:
        cp = (beta, Sigma, gamma1, gamma2, nu)
        
    return(cp)
       
  
def mst_mardia(delta_sq, nu, d): 
    
    """
    Mardia measures gamma1 and gamma2 for MST; book: (6.31), (6.32), p.178
    """
    
    if (d < 1):
        raise(ValueError("d < 1")) 
    if (d != round(d)): 
        raise(ValueError("'d' must be a positive integer"))
    if ((delta_sq < 0) or (delta_sq > 1)):  
        raise(ValueError("delta.sq not in (0,1)"))
    if (nu <= 3): 
        raise(ValueError("'nu>3' is required"))
        
    cumul = st_cumulants(0, 1, np.sqrt(np.divide(delta_sq,(1-delta_sq))), nu).reshape((4,))
    mu = cumul[0]
    sigma = np.sqrt(cumul[1])
    gamma1 = np.divide(cumul[2],np.power(sigma,3))
    gamma2 = np.divide(cumul[3],np.power(sigma,4))
    if (nu > 3):
        gamma1M = np.square(gamma1) + 3*(d-1)*np.divide(np.square(mu),\
                             np.multiply((nu-3),np.square(sigma))) 
    else: 
        gamma1M = np.infty
    are = lambda nu, k1, k2: 1/(1 - k2/nu) - k1/(nu - k2) # (nu-k1)/(nu-k2)
    if (nu > 4):
        gamma2M = (gamma2 + 3 +(d**2-1)*are(nu,2,4) + 2*(d-1)*(are(nu,0,4)
                -np.square(mu)*are(nu,1,3))/np.square(sigma) - d*(d+2))
    else: 
        gamma2M = np.infty
        
    return(gamma1M,gamma2M)            
    
  
def st_gamma1(delta, nu):
    
    """
    Skew-t gamma 1 measure.
    Remark: this function is vectorized w.r.t. delta, but takes a single value of nu
    """
    
    if type(nu) == np.ndarray:
        if (nu.shape[0] > 1): 
            raise(ValueError("'nu' must be a single value"))
        nu = nu[0]
    if (nu <= 0): 
        raise(ValueError("'nu' must be positive"))
    out = np.repeat(np.nan, len(delta)) 
    ok = (np.abs(delta) <= 1) 
    if ((nu >= 3) and (np.sum(ok) > 0)): 
        alpha = np.divide(delta[ok],np.sqrt(1 - np.square(delta[ok])))
        cumul = st_cumulants(0, 1, alpha, nu, n=3) 
        if (np.sum(ok) == 1):
            out[ok] = cumul[2]/cumul[1]^1.5 
        else: 
            out[ok] = np.divide(cumul[:,2],np.power(cumul[:,1],1.5))  
    return(out) 

      
def st_cumulants(xi=0, omega=1, alpha=0, nu=np.infty, n=4):
    
    """
    Calculates the skew T cumulants up to order 4 for given distribution parameters
    and degrees of freedom 
    """
    
    if type(nu) == np.ndarray:
        if (nu.shape[0] > 1): 
            raise(ValueError("'nu' must be a single value"))
        nu = nu[0]
    if type(alpha) == np.ndarray:
        d = alpha.shape[0]    
    elif type(alpha) in (int,float,np.int_,np.float_):
        d = 1
    else:
        raise(ValueError("Please provide xi as scalar or array"))
    if(np.isinf(nu)): 
        # return(sn.cumulants(xi, omega, alpha, n=n))
        raise(NotImplementedError("At nu=inf, sn cumulants ought to be returned. To be implemented"))
    n = min((n,4))      
    #  par = cbind(xi, omega, alpha)
    #  alpha <- par[,3]
    if (np.abs(alpha) < np.infty).all():
        delta = np.divide(alpha,np.sqrt(1+np.square(alpha)))
    else:
        delta = np.sign(alpha)
    cumul = np.full((d,n),np.nan)
    mu = np.multiply(bleat(nu),delta)
    cumul[:,0] = mu
    # r <- function(nu, k1, k2) 1/(1-k2/nu) - k1/(nu-k2)     # = (nu-k1)/(nu-k2)
    s = lambda nu, k: np.divide(1,(1 - np.divide(k,nu)))                        # = nu/(nu-k)
    if ((n>1) and (nu>2)): 
        cumul[:,1] = s(nu,2) - np.square(mu)
    if ((n>2) and (nu>3)): 
        cumul[:,2] = np.multiply(mu,(np.multiply(3-np.square(delta),s(nu,3)) - 3*s(nu,2) + 2*np.square(mu)))
    if ((n>2) and (nu==3)): 
        cumul[:,2] = np.sign(alpha) * np.infty  
    if ((n>3) and (nu>4)): 
        cumul[:,3] = 3*np.multiply(s(nu,2),s(nu,4)) \
                    - 4*np.multiply(np.square(mu),np.multiply(3-np.square(delta),s(nu,3))) \
                    + 6*np.multiply(np.square(mu),s(nu,2))-3*np.power(mu,4) \
                    - 3*np.square(cumul[:,1])
    if ((n>3) and (nu==4)): 
        cumul[:,3] = np.infty
    cumul = np.multiply(cumul,np.power.outer(omega,np.arange(1,n+1)))
    cumul[:,0] = cumul[:,0] + xi
    
    return(cumul)
  
    
def bleat(nu):  
    """
    function b(.) in SN book, eq.(4.15)
    longer name picked to avoid confusion
    vectorized for 'nu', intended for values nu>1, otherwise it returns NaN
    """  

    if type(nu) in (int,float,np.int_,np.float_):
        nu = np.array([nu])
    out = np.repeat(np.nan, len(nu))
    big = (nu > 1e4)
    ok = np.where(np.logical_and(np.logical_and(nu > 1, nu <= 1e4),np.equal(np.isnan(nu),False)))
    # for large nu use asymptotic expression (from SN book, exercise 4.6)
    out[big] = np.sqrt(2/np.pi)*(1+np.divide(0.75,nu[big])+np.divide(0.78125,np.square(nu[big])))
    out[ok] = np.multiply(np.sqrt(nu[ok]/np.pi),
       np.exp(sps.gammaln((nu[ok]-1)/2) - sps.gammaln(nu[ok]/2))
       )
    return(out)
  

def msn_dp2cp(xi,Omega,alpha,tau=0, aux=False): 
    
    """
    Transformation of direct distribution parameters to centred distribution 
    parameters for the skewnormal family. 
    Inputs: 
        xi, (n,), (n,1) or (1,n) matrix or array, location
        Omega, (n,n), matrix, Scale
        alpha, (n,), (n,1) or (1,n) matrix or array, skewness
        tau, (n,), (n,1) or (1,n) matrix or array, tau parameter(s)
        aux, bool, to report auxiliary estimates. 
    Output
        a tuple comtaining: transformed location, scale, skewness and input skewness
    Remark
        Works for univariate parameters, but they have to be entered as array 
        or matrix, e.g. 
        msn_dp2cp(np.array([1]),np.array([2]),np.array([1]))
    """
    
    xi = _check_format(xi)
    alpha = _check_format(alpha)
    tau = _check_format(tau)
        
    d = alpha.shape[0]
#    if type(tau) in [int,float,np.int_,np.float_]:
#        tau = np.repeat(tau,d)
#    
#    if (type(tau)==np.ndarray and (tau.shape in [(1,),(1,0),(1,1)])):
#        tau = float(tau)
#        tau = np.repeat(tau,d)
        
    Omega = np.matrix(Omega)  #maybe not necessary 
    omega = np.sqrt(np.diag(Omega))
    delta,alpha_star,delta_star,Ocor = delta_etc(alpha, Omega)
    mu_z = np.multiply(zeta(1, tau),delta)
    sd_z = np.sqrt(1 + np.multiply(zeta(2, tau),np.square(delta)))
    Sigma = Omega + np.multiply(zeta(2,tau),np.outer(np.multiply(omega,delta), np.multiply(omega,delta)))
    Sigma[np.tril_indices(d,k=-1)] = Sigma[np.triu_indices(d,k=1)] #only lower trianguilar in Azzalini
    gamma1 = np.multiply(zeta(3, tau),np.power(np.divide(delta,sd_z),3))
    if (type(alpha) == np.ndarray): #multivariate
        beta = xi + np.multiply(mu_z,omega)
        cp = (beta, Sigma, gamma1, tau)
    else: #univariate
        beta = xi + np.multiply(mu_z,omega)
        cp = (beta, Sigma, gamma1, tau)
    if(aux):
        if len(delta.shape)>1:
            if delta.shape[1] > delta.shape[0]:
                delta = np.array(delta).reshape(-1)
        lambhdha = np.divide(delta,np.sqrt(1-np.square(delta)))
        D = np.diag(np.sqrt(1+np.square(lambhdha)))
        Ocor = cov2cor(Omega)
        Psi = np.matmul(D,np.matmul((Ocor-np.outer(delta,delta)),D))
        Psi = (Psi + Psi.T)/2
        O_inv = np.linalg.inv(Omega)
        O_pcor = -cov2cor(O_inv) 
        O_pcor[np.diag_indices(d,ndim=2)] = 1 
        R = Ocor + np.multiply(zeta(2,tau),np.outer(delta,delta))
        R[np.tril_indices(d,k=-1)] = R[np.triu_indices(d,k=1)]
        ratio2 = np.divide(np.square(delta_star),1+np.multiply(zeta(2,tau),np.square(delta_star)))
        gamma1M = np.multiply(np.square(zeta(3,tau)),np.power(ratio2,3)) 
        gamma2M = np.multiply(zeta(4,tau),np.square(ratio2))
        # SN book: see (5.74), (5.75) on p.153
        cp = (beta, Sigma, gamma1, tau, omega, R, O_inv, Ocor, O_pcor, 
              lambhdha, Psi, delta, delta_star, alpha_star, gamma1M, gamma2M)
    return(cp) 

  
def delta_etc(alpha,*args):
    
    """
    alpha = location paramter
    Scale parameter can be passed as optional argument
    """
    
    largs = len(args)
    if np.isinf(alpha).any():
        inf = np.where(np.isinf(np.abs(alpha)))
        inf_flag = True
    else:
        inf_flag = False
        
    if type(alpha)==np.matrix:
        alpha = np.array(alpha)
        
    if len(alpha.shape) > 1: 
        alpha = alpha.reshape(-1)
    
    if(largs == 0): # case d=1
        delta = alpha/np.sqrt(1+np.square(alpha))
        if inf_flag:
            delta[inf] = np.sign(alpha[inf])
        alpha_star = np.nan 
        delta_star = np.nan
        Ocor = np.nan
    else: # d>1
        Omega = args[0]
        if(any(Omega.shape != np.repeat(len(alpha),2))): 
            raise(ValueError("Dimension mismatch"))
        Ocor = cov2cor(Omega)
        if(not(inf_flag)): # d>1, standard case
            Ocor_alpha = np.matmul(Ocor,alpha)
            alpha_sq = np.sum(np.multiply(alpha,Ocor_alpha))
            delta = Ocor_alpha/np.sqrt(1+alpha_sq)
            alpha_star = np.sqrt(alpha_sq)
            delta_star = np.sqrt(alpha_sq/(1+alpha_sq))

        else: # d>1, case with some abs(alpha)=Inf
            if(len(inf) > 1): 
                warnings.warn("Several abs(alpha)==Inf, I handle them as 'equal-rate Inf'") 
            k = np.repeat(0,alpha.shape[0])
            if inf_flag:
                k[inf] = np.sign(alpha[inf])
            Ocor_k = np.matmul(Ocor,k) 
            delta = Ocor_k/np.sqrt(np.sum(np.multiply(k,Ocor_k)))
            delta_star = 1
            alpha_star = np.infty
    return(delta, alpha_star, delta_star, Ocor)
    
     
def cov2cor(Sigma):
    
    """
    Efficient transformation of covariance into correlation matrix
    """
    
    n,p = Sigma.shape
    if (p != n): 
        raise(ValueError("'Sigma' must be a square numeric matrix"))
    Is = np.sqrt(1/np.diag(Sigma))
    if (not(np.isfinite(Is)).any): 
        warnings.warn("diag(.) had 0 or NA entries; non-finite result is doubtful")
    Rho = Sigma
    Rho = np.multiply(np.multiply(Is,Sigma),np.repeat(Is,p).reshape((p,p)))
    Rho[np.diag_indices(p,ndim=2)] = 1
    return(Rho)
 
    
def force_symmetry(x, tol=10*np.sqrt(np.finfo(float).resolution)): 
    
    if (type(x) not in (np.matrix,np.ndarray)): 
        raise(ValueError("x must be a matrix"))
    err = np.divide(np.abs(x-x.T),(1+np.abs(x)))
    max_err = np.max(np.divide(err,(1+err)))
    if max_err > tol: 
        warnings.warn("matrix seems not to be symmetric")
    if max_err > 100*tol: 
        raise(ValueError("Matrix is not symmetric"))
    return((x + x.T)/2)

    
def zeta(k, x): # k integer in (0,5)
    
    if type(x) == np.matrix:
        x = np.array(x)
        
    if len(x.shape) > 1:
        x = x.reshape(-1)
  
    if(((k<0 or k>5) or not(k == round(k)))): 
        return(None)
        
    na = np.isnan(x)
    if na.any():
        x[na] = 0
    x2 = np.square(x)
    if k==0:
        z = np.log(spp.norm.cdf(x)) + np.log(2)
    if k==1:
        ind_sm_neg_50 = (x <= -50)
        z = x
        if ind_sm_neg_50.any():
            xx = x[ind_sm_neg_50]
            xx2 = x2[ind_sm_neg_50]
            z[ind_sm_neg_50] = -np.divide(xx,1 -np.divide(1,(xx2+2)) +np.divide(1,np.multiply((xx2+2),(xx2+4))) 
                -np.divide(5,np.multiply(np.multiply(xx2+2,xx2+4),(xx2+6)))
                +np.divide(9,np.multiply(np.multiply(np.multiply(xx2+2,xx2+4),xx2+6),xx2+8)) 
                -np.divide(129,np.multiply(np.multiply(np.multiply(np.multiply(xx2+2,xx2+4),xx2+6),xx2+8),xx2+10)))  
            z[not(ind_sm_neg_50)] = np.exp(np.log(spp.norm.pdf(x[not(ind_sm_neg_50)])) - np.log(spp.norm.cdf(x[not(ind_sm_neg_50)])))
        else:
            z = np.exp(np.log(spp.norm.pdf(x)) - np.log(spp.norm.cdf(x)))
    if k==2:
        z = -np.multiply(zeta(1,x),x+zeta(1,x))
    if k==3:
        z = -np.multiply(zeta(2,x),x+zeta(1,x)) - np.multiply(zeta(1,x),1+zeta(2,x))
    if k==4:
        z = -np.multiply(zeta(3,x),x+2*zeta(1,x)) - 2*np.multiply(zeta(2,x),1+zeta(2,x))
    if k==5:
        z = -np.multiply(zeta(4,x),x+2*zeta(1,x)) -np.multiply(zeta(3,x),3+4*zeta(2,x)) \
            -2*np.multiply(zeta(2,x),zeta(3,x))
    
    neg_inf = (x == -np.inf)
    if(neg_inf.any()):
        if k==1:
            z[neg_inf] = np.inf
        if k==2:
            z[neg_inf] = -1
        if k in (3,4,5):
            z[neg_inf] = 0
    
    pos_inf = (x==np.inf)        
    if ((k>1) and pos_inf.any()): 
        z[pos_inf] = 0
        
    return(z)