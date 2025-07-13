from scipy.stats import nbinom, poisson
import numpy as np
import scipy.stats as stats

def add_fit_method(distribution_instance):
    def poisson_fit(self):
        """
        Fit the Poisson distribution to data.
        
        Parameters:
        - data: array-like of observed counts
        
        Returns:
        - mu: estimated mean (lambda) of the distribution.
        """
        # Estimate λ (mean of the data)
        mu_estimate = np.mean(self.data)
        return (mu_estimate,)
    
    # Add fit methods to the distribution instance
    setattr(distribution_instance, 'fit', poisson_fit)
    
    return distribution_instance

# Apply the decorator directly to the poisson instance
poisson = add_fit_method(poisson)

def add_fit_method(distribution_instance):
    def nbinom_fit(self):
        """
        Fit the negative binomial distribution to data.
        
        Parameters:
        - data: array-like of observed counts
        
        Returns:
        - (n, p): estimated parameters where n is the number of successes and p is the probability of success.
        """
        mean = np.mean(self.data)
        variance = np.var(self.data)
        
        if variance <= mean:
            raise ValueError("Variance must be greater than the mean for a negative binomial distribution.")
        
        p = mean / variance
        n = mean ** 2 / (variance - mean)
        
        return (n, p)
    
    # Add the fit methods to the distribution instance
    setattr(distribution_instance, 'fit', nbinom_fit)
    
    return distribution_instance

# Apply the decorator directly to the nbinom instance
nbinom = add_fit_method(nbinom)

'''
def modify_scipy_distribution(distribution_class, actuarial_name):
    """
    Directly modifies a given scipy.stats distribution instance to:
    1. Accept input parameters in actuarial form.
    2. Return fitted parameters in actuarial form with loc=0.

    Parameters:
    - distribution_class: SciPy distribution instance (e.g., stats.gamma)
    - actuarial_name: Actuarial distribution name
    """

    original_fit = distribution_class.fit  # Preserve original fit function
    original_rvs = distribution_class.rvs  # Preserve original random sampling

    def actuarial_rvs(*args, **kwargs):
        """Generate random variables using actuarial parameterization."""
        if actuarial_name == "gamma":
            alpha, theta = args[:2]
            return original_rvs(a=alpha, scale=theta, **kwargs)

        elif actuarial_name == "weibull_min":
            alpha, beta = args[:2]
            return original_rvs(c=alpha, scale=beta, **kwargs)

        elif actuarial_name == "pareto":
            alpha, theta = args[:2]
            return original_rvs(b=alpha, scale=theta, **kwargs)

        elif actuarial_name == "lognorm":
            mu, sigma = args[:2]
            return original_rvs(s=sigma, scale=np.exp(mu), **kwargs)

        elif actuarial_name == "beta":
            alpha, beta = args[:2]
            return original_rvs(a=alpha, b=beta, **kwargs)

        return original_rvs(*args, **kwargs)

    def actuarial_fit(data, *fit_args, **fit_kwargs):
        """Modified fit method that returns actuarial-style parameters."""
        fit_kwargs["floc"] = 0  # Force loc=0 for all distributions
        fitted_params = original_fit(data, *fit_args, **fit_kwargs)

        if len(fitted_params) == 3:  # Standard SciPy format (shape, loc, scale)
            shape, loc, scale = fitted_params
        elif len(fitted_params) == 2:  # Sometimes loc is omitted
            shape, scale = fitted_params
            fitted_params = shape, 0, scale  # Add loc=0
        if actuarial_name == "gamma":
            alpha, loc, theta = fitted_params
            return alpha, theta  # Drop loc (assumed 0)

        elif actuarial_name == "weibull_min":
            alpha, loc, beta = fitted_params
            return alpha, beta  # Drop loc

        elif actuarial_name == "pareto":
            alpha, loc, theta = fitted_params
            return alpha, theta  # Drop loc

        elif actuarial_name == "lognorm":
            sigma, loc, scale = fitted_params
            mu = np.log(scale) # Convert scale back to mu
            return mu, sigma  # Drop loc

        return shape, loc, scale  # Default case

    # Create a new frozen distribution with modified methods
    modified_distribution = distribution_class
    modified_distribution.rvs = actuarial_rvs
    modified_distribution.fit = actuarial_fit

    return modified_distribution

# Apply modifications directly to distribution instances
stats.gamma = modify_scipy_distribution(stats.gamma, "gamma")
stats.lognorm = modify_scipy_distribution(stats.lognorm, "lognorm")  
stats.pareto = modify_scipy_distribution(stats.pareto, "pareto")
stats.weibull_min = modify_scipy_distribution(stats.weibull_min, "weibull_min")
'''
if __name__ == "__main__":
    dist = stats.lognorm(scale = np.exp(0.5), s = 0.2).rvs(size=1000)
    # Testing
    dist = stats.lognorm.rvs(0.5, 0.2, size=1000)
    dist.mean()
    dist.std()
    dist.mean()
    fitted_params = stats.lognorm.fit(dist)
    dist = stats.lognorm(-0.024008833369301152, 0.5125642303586879)
    dist.rvs(size=1000)