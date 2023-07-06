from enum import Flag
from glob import glob
from os import stat
import random
from turtle import shape
import numpy as np
from scipy.stats import norm
from scipy.stats import gamma
from distfit import distfit

# These variables should never be imported from this file.
_samples = {}
"""Stores a dictionary of the form {key: list of distribution samples}"""
_invcdfs = {}
"""Stores a dictionary of the form {key: list of inverse cdf points}"""
_empiricalData = {}
"""Stores a dictionary of the form {key: list of empirical data}"""

MAX_RESAMPLE = 10

def collect_data(rundir):
    """Unimplemented function intended for empirical data collection"""
    pass

def empirical_sample(distribution_name, state=None) -> float:
    """Gets a sample from a specified distribution.

    Return:
        Returns a float from the distribution.
    """
    dist, loc, par, res, neg = distribution_name

    if distribution_name in _samples:
        """if a distribution is already in the global samples, 
            we draw a sample based on the probability in the specific distribution"""
        if state is None:
            return random.choices(_samples[distribution_name][0], weights = _samples[distribution_name][1])[0]
        return state.choice(a = _samples[distribution_name][0], weights = _samples[distribution_name][1])
    else:
        """if the wanted sample's distribution is not in _samples, we generate a single sample based on the distribution
            without adding it into _samples"""
        if state is None:
            if dist == 'gamma': return gamma_sample(loc, par)
            if dist == 'norm': return norm_sample(loc, par)
            return 'not yet'

def empirical_samples(distribution_name, size):
    """return a list of #size of samples based on the distribution specified and store the samples globally
    """
    global _empiricalData
    sampleX = []
    if distribution_name in _empiricalData:
        return np.random.choice(_empiricalData[distribution_name], size=size)
    for x in range(size):
        sampleX.append(empirical_sample(distribution_name))
    _empiricalData[distribution_name] = sampleX
    return sampleX

def generate_data(distributions:list, sizes:list):
    """return a dictionary of data acording to the distribution and size requirements
    """
    data = {}
    for distribution_name in distributions:
        indexDist = distributions.index(distribution_name)
        newdata = empirical_samples(distribution_name, sizes[indexDist])
        data[distribution_name] = newdata
    return data

def fitdist(datasets:list, sizes:list, gammaFlag=True, debug = False, types='popular', exampleDists =[], exampleSizes=[], plot=False):
    """return a dictionary of distributions of a data set and new samples drawn based on the best fit distribution
        debug: test fit against a known distribution
    """
    dist = distfit(distr=types)  
    fits = {}
    example = exampleDists[0]  

    if debug:
        for distribution_name in exampleDists:
            indexDist = exampleDists.index(distribution_name)
            X = np.array(empirical_samples(distribution_name, exampleSizes[indexDist] ))
            dist.fit_transform(X)
            guessDist = dist.model['name']
            fig, ax = dist.plot()
            fig.show()
            summary, ax2 = dist.plot_summary()
            summary.show()
            fits[distribution_name] = (guessDist, dist.model)
    else:
        for data in datasets:
            if type(data) == float or type(data) == int:
                data = [data]
            indexData = datasets.index(data)
            if sizes[indexData] == len(data):
                X = np.array(data)
            else:
                X = np.array(np.random.choice(a = data, size = sizes[indexData]))
            dist.fit_transform(X, verbose=1)
            if dist.model['name'] == 'gamma':
                if dist.model['arg'][0] > 40 and not gammaFlag: # alpha is too big so we can treat the distribution as approxiamtely normal
                    dist = distfit(distr='norm')
                    dist.fit_transform(X, verbose=1)
                else:
                    while (dist.model['arg'][0]) > 40: # we would want a smaller gamma
                        X = np.array(np.random.choice(a = data, size = sizes[indexData]))
                        dist.fit_transform(X, verbose=1)
            guessDist = (dist.model['name'], dist.model['loc'] if dist.model['name'] =='norm'else dist.model['arg'][0], dist.model['scale'] if dist.model['name'] =='norm' else (1/dist.model['scale'], dist.model['loc']), len(data), False)
            if plot:
                fig, ax = dist.plot()

                dist_edge = example[0]
                a = example[1]
                par = example[2]
                print(dist, a, par)
                if dist_edge == 'G':
                    x,y = gamma_curve(alpha = float(a), beta = eval(par)[0])
                    x = [loc + eval(par)[1] for loc in x]
                elif dist_edge == 'N':
                    x,y = norm_curve(mu = float(a), sigma = float(par))
                ax.plot(x, y, 'g', linewidth=1, label='acutal distribution')
                ax.legend()
                fig.show()
            fits[guessDist] = (dist.model)
            
    return fits

def predict_sample(distribution_name, data:list, plot=False):
    """return a list of probs for data based on the specified distribution
    """
    dist = distfit()
    
    pool = empirical_samples(distribution_name, size=1000)
    dist.fit_transform(np.array(pool))
    results = dist.predict(data)
    if plot:
        fig, ax = dist.plot()
        fig.show()
    return results

def gamma_sample(alpha: float, beta:float, state = None, res=1000, neg=False) -> float:
    """return a gamma sample based on shape and scale
    """
    count = 0
    ans = -1.0
    theta = 1/beta
    while ans < 0.0:
        if count > MAX_RESAMPLE:
            ans = 0.0
            break
        if state is None:
            ans = np.random.gamma(shape=alpha, scale=theta, size=None)
        else:
            ans = state.gamma(shape=alpha, scale=theta, size=None)
        count += 1
    
    return ans

def norm_sample(mu: float, sigma: float, state=None, res=1000,
                neg=False) -> float:
    """Retrieve a sample from a normal distribution

    Args:
        mu: mean of the normal distribution
        sigma: standard deviation of the normal distribution

    Keyword Args:
        mu: Mean of the normal curve
        sigma (float): Standard Dev of the normal curve
        state (RandomState, optional): Numpy RandomState object to use for the
            sampling. Default is set to None, meaning that we're using the
            global state.
        res (int, optional): Resolution
        neg (bool, optional): If we want to use negative values, set this to
            True. Default is False.
    Return:
        Returns a random sample (float).
    """
    ans = -1.0
    count = 0
    while ans < 0.0:
        if count > MAX_RESAMPLE:
            ans = 0.0
            break
        if state is None:
            ans = np.random.normal(loc=mu, scale=sigma)
        else:
            ans = state.normal(loc=mu, scale=sigma)
        count += 1
    return ans

def gamma_curve(alpha: float, beta: float, res = 1000, neg=False):
    """gamma pdf curve
    """
    global _samples
    theta = 1/beta
    if ('gamma', alpha, beta, res, neg) in _samples:
        return _samples[('gamma', alpha, beta, res, neg)]
    if neg:
        theta = 1/beta
        x = np.linspace(gamma.ppf(0.003, a=alpha, scale=theta),
                        gamma.ppf(0.997, a=alpha, scale=theta),
                        res)
    else:
        x = np.linspace(max(gamma.ppf(0.003, a=alpha, scale=theta), 0.0),
                        max(gamma.ppf(0.997, a=alpha, scale=theta), 0.0),
                        res)
    y = gamma.pdf(x, alpha, scale=theta)
    _samples[(('gamma', alpha, beta, res, neg))] = (x,y)

    return(x,y)

def norm_curve(mu: float, sigma: float, res=1000, neg=False):
    """Produces a descritised normal curve.

    Note:
        This function is memoised. We don't want to redo work that we've
        already done.

    Example:
        norm_curve(1.0, 0.0)
    """
    global _samples

    # Memoisation check
    if (mu, sigma, res, neg) in _samples:
        return _samples[('norm', mu, sigma, res, neg)]
    if neg:
        x = np.linspace(norm.ppf(0.003, loc=mu, scale=sigma),
                        norm.ppf(0.997, loc=mu, scale=sigma),
                        res)
    else:
        x = np.linspace(max(norm.ppf(0.003, loc=mu, scale=sigma), 0.0),
                        max(norm.ppf(0.997, loc=mu, scale=sigma), 0.0),
                        res)
    y = norm.pdf(x, loc=mu, scale=sigma)
    # Memoisation
    _samples[('norm', mu, sigma, res, neg)] = (x, y)
    return (x, y)

def invcdf_gamma_curve(alpha:float, beta:float, res=1000, neg=False):
    """Generate an inverse CDF curve for a gamma distribution
    """
    global _invcdfs
    if ('gamma', alpha, beta, res, neg) in _invcdfs:
        return _invcdfs[('gamma', alpha, beta, res, neg)]
    gammax, gammay = gamma_curve(alpha, beta, res=res, neg=neg)
    delx = gammax[1] - gammax[0]
    sol = (np.cumsum(gammay) * delx, gammax)
    _invcdfs[('gamma', alpha, beta, res, neg)] = sol
    return sol

def invcdf_norm_curve(mu: float, sigma: float, res=1000, neg=False):
    """Generate an inverse CDF curve for a normal distribution
    """
    global _invcdfs
    if ('norm', mu, sigma, res, neg) in _invcdfs:
        return _invcdfs[('norm', mu, sigma, res, neg)]
    normx, normy = norm_curve(mu, sigma, res=res, neg=neg)
    delx = normx[1] - normx[0]
    sol = (np.cumsum(normy) * delx, normx)
    _invcdfs[('norm', mu, sigma, res, neg)] = sol
    return sol

def binary_search_lookup(val, l):
    """Returns the index of where the val is in a sorted list.

    If val is not in l, then return the index of the element directly below l.

    Example:
        >>> binary_search_lookup(10.0, [-5.0, 6.0, 10.0, 100.0])
        2
        >>> binary_search_lookup(5.0, [-5.0, 4.0, 10.0, 100.0])
        1
        >>> binary_search_lookup(11.0, [-5.0, 4.0, 10.0, 100.0])
        2
    """
    up = len(l) - 1
    lo = 0
    look = (up + lo) // 2
    while abs(up - lo) > 1:
        if l[look] == val:
            return look
        if val < l[look]:
            # We need to look lower.
            up = look
            look = (up + lo) // 2
        else:
            # We need to look higher.
            lo = look
            look = (up + lo) // 2
    # Didn't find the exact match, return the lower bound then.
    return lo

def invcdf_gamma(val: float, alpha: float, beta:float, res=1000, neg=False):
    curve = invcdf_gamma_curve(alpha, beta, res=res, neg=neg)
    ans = curve[1][binary_search_lookup(val, curve[0])]
    return ans

def invcdf_norm(val: float, mu: float, sigma: float, res=1000, neg=False):
    """Returns the inverse cumulative density function for a normal curve

    Args:
        val (float): input (x-axis) for the inverse CDF.
        mu (float): mean of the normal curve.
        simga (float): sd of the normal curve.
        res (int, optional): resolution of the normal curve.
        neg (bool, optional): Should include negative values in the cdf.
    """
    # functiontimer.start("invcdf_norm")
    curve = invcdf_norm_curve(mu, sigma, res=res, neg=neg)
    ans = curve[1][binary_search_lookup(val, curve[0])]
    # functiontimer.stop("invcdf_norm")
    return ans


def uniform_sample(lb: float, ub: float, random_state=None) -> float:
    """Returns a randomly selected uniform sample

    Args:
        lb: Lower bound of the uniform sample
        ub: Upper bound of the uniform sample
        random_state (optional): Numpy random state to use (default None)

    Return:
        A new sample drawn from the uniform distribution
    """
    if random_state is None:
        return np.random.uniform(lb, ub)
    else:
        return random_state.uniform(lb, ub)

def invcdf_uniform(val: float, lb: float, ub: float) -> float:
    """Returns the inverse CDF lookup of a uniform distribution. Is constant
        time to call.

    Args:
        val: Value between 0 and 1 to calculate the inverse cdf of.
        lb: lower bound of the uniform distribution
        ub: upper bound of the uniform distribution

    Returns:
        Inverse CDF of a uniform distribution for the provided value.

    Examples:
        >>> invcdf_uniform(1, 5, 10)
        10
        >>> invcdf_uniform(0, 5, 10)
        5
        >>> invcdf_uniform(0.5, 5, 10)
        7.5
    """
    if val < 0:
        return -float("inf")
    elif val > 1:
        return float("inf")
    else:
        return val * (ub - lb) + lb


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    import matplotlib.pyplot as plt
    curve = invcdf_norm_curve(1, 1)
    data = list(generate_data([('norm', 20, 2, 1000, False)], [1000]).values())[0]
    fits = fitdist([data], [1000], [0], plot=True, types = 'gamma')
    # curve = gamma_curve(1, 1)    ##gamma pdf

    # curve = invcdf_gamma_curve(5, 1)
    # plt.plot(curve[0], curve[1])
    # plt.show()
