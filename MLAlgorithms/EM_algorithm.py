import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

__author__ = 'Ranel Karimov'

if __name__ == '__main__': 
    data = np.loadtxt("data/em_data.txt")
    x_train, x_test = train_test_split(data, test_size=0.4, train_size=0.6)
    model = GaussianMixture(2, tol=0.0001, covariance_type='spherical')
    model.fit(x_train)
    model.predict(x_test)
    
    # Create plot 
    plt.title('Gaussian Mixture Model')

    plt.gcf().set_size_inches(8, 8)
    plt.xlim(np.min(data[:, 0]) - .5, np.max(data[:, 0]) + .5) 
    plt.ylim(np.min(data[:, 1]) - .5, np.max(data[:, 1]) + .5)
    # Get learned model parameters.
    means, covs = model.means_, model.covariances_ 
    for k in range(2):
        # Visualize means.
        plt.scatter( means[k, 0], means[k, 1], color='k', alpha=0.7, s=150, marker='X')
        # Visualize covarianes 
        for fac in range(1, 4): 
            circle = plt.Circle(means[k], fac*np.sqrt(covs[k]), color='k', fill=False)
            
    plt.gca().add_artist(circle)
    # Visualize data points.
    plt.scatter(data[:, 0], data[:, 1], alpha=.4) 
    plt.show()