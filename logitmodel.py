import numpy as np 
from scipy.optimize import minimize
import warnings
from scipy.special import logsumexp
import pandas as pd 

def get_Xbeta(x,beta):
    """Omit beta_0 (i.e. parameter vector for the first choice option)
    from beta array. beta_0 is set to the zero vector to avoid indeterminacy."""

    n,k = x.shape
    J = len(beta)//k
    Xb = np.zeros((n,J+1))
    for j in range(J):
        Xb[:,j+1] = x@beta[j*k:(j+1)*k]

    return Xb 


def get_Dij(choices,J):
    dij = np.zeros((choices.shape[0],J))
    dij[np.arange(choices.shape[0]),choices] = 1
    return dij 


def get_Probs(x,beta):
    logP = get_logProbs(x, beta)
    return np.exp(logP) 


def get_logProbs(x,beta):
    Xb = get_Xbeta(x, beta)
    logSeXb = logsumexp(Xb,axis=1,keepdims=True)
    logP = Xb - logSeXb 
    return logP


def get_negjac(beta,x,choices):
    """The first derivative of the negative loglikelihood w.r.t. 
    the model parameters."""

    n,k = x.shape
    J = len(beta)//k  
    dij = get_Dij(choices,J+1) 
    P = get_Probs(x, beta) 

    dijP = dij - P
    jac = np.zeros(J*k)
    for i in range(J):
        j = i+1
        jac[i*k:(i+1)*k] = dijP[:,j][None,:]@x
         
    return -jac


def get_neghess(beta,x,*args):
    """The matrix of second derivatives of the negative loglikelihood w.r.t. 
    the model parameters."""

    n,k = x.shape
    J = len(beta)//k  
    P = get_Probs(x, beta) 

    hess = np.zeros((J*k,J*k))
    for j in range(J):
        for l in range(J):
            d = 1 if j==l else 0 
            pp = P[:,j+1]*(d-P[:,l+1])
            hess[j*k:(j+1)*k,l*k:(l+1)*k] = (pp[:,None]*x).T@x

    return hess 


def negloglike(beta,x,choices):
    n = x.shape[0]
    idx = np.arange(n)
    logP = get_logProbs(x, beta)
    return -np.sum(logP[idx,choices])



def fit(choices,exog,b0,**kwargs):

    res = minimize(negloglike, b0, 
        args=(exog,choices),
        **kwargs
    )
    
    return res 


def cov(beta,x):
    negH = get_neghess(beta, x)
    return np.linalg.inv(negH)


def get_marginaleff(x,beta):

    n,k = x.shape
    if k==0:
        x = x[None,:]
        n,k = x.shape
        
    P = get_Probs(x, beta)
    J = P.shape[1]
    _beta = np.append(np.zeros(k),beta)
    meanbeta = P@_beta.reshape((J,k)) 

    me = P.repeat(k,axis=1)*(_beta - np.tile(meanbeta,J))
    me = np.mean(me,axis=0).reshape((J,k))
    return me


def simulate_data(n=1000,all_chosen_once=True,max_iter=100):

    n_choices = 3
    k = 4

    x = np.ones((n,k))
    count = 0
    while True:

        x[:,1] = np.random.gamma(7.5,1,size=n)
        x[:,2] = np.random.choice(2,size=n,p=[0.7,0.3])
        x[:,3] = x[:,1]*x[:,2] 

        beta = np.random.uniform(-1,1,size=k*(n_choices-1))

        cumProb = np.cumsum(get_Probs(x, beta),axis=1) 
        u = np.random.rand(n)
        choices = np.zeros(n,dtype=int)
        for i in range(n_choices-1):
            choices[(u>cumProb[:,i]) & (u<=cumProb[:,i+1])] = i+1

        c, c_count = np.unique(choices,return_counts=True)
        if not all_chosen_once:
            break
        elif len(c)==n_choices and np.all(c_count/np.sum(c_count)>=0.1):
            break 
        elif count >= max_iter:
            msg = "maximum number iterations reached without choice vector incorporating all possible choise."
            warnings.warn(msg) 

        count+=1

    return choices,x,beta

#%%
if __name__=="__main__":


    y,exog,b_true = simulate_data(10000)
    
    kwargs = {
        'method': 'L-BFGS-B',
        'jac':get_negjac,
        'hess':get_neghess
    }

    res = fit(y,exog,10*np.random.uniform(-1,1,size=b_true.shape),
        **kwargs
    ) 
    for i in range(100):
        res_i = fit(y,exog,np.random.uniform(-1,1,size=b_true.shape),
            **kwargs
        )
        if res_i.success and res_i.fun<res.fun:
            res = res_i

    
    me = get_marginaleff(exog, res.x)
    se = np.sqrt(np.diag(cov(res.x,exog)))
    print(me)
    print(
        pd.DataFrame({
            'b': b_true,
            'b_hat': res.x,
            'se': se
            }).round(4)
    ) 


    # empirical replication
    # see:
    # - R: https://stats.idre.ucla.edu/r/dae/multinomial-logistic-regression/
    # - Stata: https://stats.idre.ucla.edu/stata/dae/multinomiallogistic-regression/ 
    df = pd.read_stata("https://stats.idre.ucla.edu/stat/data/hsbdemo.dta")

    exog = df[['write']].merge(pd.get_dummies(df['ses']),right_index=True,left_index=True,how='inner')
    exog['const'] = np.ones(len(df.index))
    exog.drop(columns='low',inplace=True)

    y = df.prog.map({'vocation':2,'general':1,'academic':0})
    
    kwargs = {
        'method': 'L-BFGS-B',
        'jac':get_negjac,
        'hess':get_neghess
    }
    res = fit(y,exog,np.random.uniform(-1,1,size=2*exog.shape[1]),
        **kwargs
    ) 

    se = np.sqrt(np.diag(cov(res.x,exog)))
    print(
        pd.DataFrame({
            'b_hat': res.x,
            'se': se
            },
            index=pd.MultiIndex.from_product([['general','vocation'],exog.columns])).round(4)
    ) 
