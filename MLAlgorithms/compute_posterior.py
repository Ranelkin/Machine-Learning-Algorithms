import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import norm

__author__ = "Ranel Karimov"

def posterior():
    """Calculating posterior distribution. 
    """
    n = 100
    x = np.linspace(-10, 10, n)
    
    mean_posterior = lambda x: x*8*np.mean(x)/(x*8+8)
    variance_posterior = lambda x : 1/(1/8+x/8)
    prior = norm.pdf(x, loc=0, scale=8)
    plt.plot(x, prior)
    posterior = lambda x, n: norm.pdf(x, loc=mean_posterior(x), scale=variance_posterior(n))
    plt.plot(x, posterior(x, n))
    scale = [10, 50, 100]
    for i in scale: 
        plt.plot(x, norm.pdf(x, loc=mean_posterior(x), scale=variance_posterior(i)))
    
    plt.figure(figsize=(8, 5))
    plt.show()
    
if __name__ == '__main__':
    posterior()