import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal

__author__ = 'Ranel Karimov'

#Own implementation 
class GMM:
    def __init__(self, num_components, max_iters=100, tol=0.0001) -> None:
        self.num_components = num_components
        self.max_iters = max_iters
        self.tol = tol
        self.x = None  
        self._weights = None  
        self.means = None    
        self.cov = None     
        self._responsibilities = None  
        self._lnp = -np.inf 
        
    def _log_likelihood(self, data, means, cov, weights):
        """Calculate log-likelihood of the data given the parameters."""
        n_samples = data.shape[0]
        log_likelihood = 0
        
        for n in range(n_samples):
            #Calculate weighted probability for each component
            weighted_probs = []
            for k in range(self.num_components):
                cov_stable = cov[k] + np.eye(cov[k].shape[0]) * 1e-6 #For num stability 
                prob = weights[k] * multivariate_normal.pdf(data[n], means[k], cov_stable)
                weighted_probs.append(prob)
            
            log_likelihood += np.log(np.sum(weighted_probs) + 1e-10)
            
        return log_likelihood
    
    def init(self, x: np.ndarray):
        """Initialize parameters"""
        self.x = x
        n_samples, n_features = x.shape
        
        indices = np.random.choice(n_samples, size=self.num_components, replace=False)
        self.means = x[indices].copy()
        
        #Assign points to nearest center
        dists = np.zeros((n_samples, self.num_components))
        for k in range(self.num_components):
            dists[:, k] = np.linalg.norm(x - self.means[k], axis=1)
        cluster_assignments = np.argmin(dists, axis=1)
        
        #Calculate init cov and weights
        self.cov = []
        self._weights = []
        
        for k in range(self.num_components):
            points_in_cluster = x[cluster_assignments == k]
            
            
            weight = len(points_in_cluster) / n_samples
            if weight == 0:
                weight = 1.0 / self.num_components
            self._weights.append(weight)
            
            if len(points_in_cluster) > 1:
                cov = np.cov(points_in_cluster, rowvar=False)
                if np.linalg.matrix_rank(cov) < n_features:
                    cov += np.eye(n_features) * 0.01
            else:
                cov = np.eye(n_features)
            self.cov.append(cov)
            
        self._weights = np.array(self._weights)
        self.cov = np.array(self.cov)
        
        self._weights = self._weights / np.sum(self._weights)
        
        self._responsibilities = np.zeros((n_samples, self.num_components))
        
        self._lnp = self._log_likelihood(x, self.means, self.cov, self._weights)
    
    def fit(self, x: np.ndarray):
        """Fit GMM using EM algorithm."""
        #Initialize parameters
        self.init(x)
        
        for iteration in range(self.max_iters):
            prev_lnp = self._lnp
            
            self._expectation()
            self._maximization()
            
            self._lnp = self._log_likelihood(self.x, self.means, self.cov, self._weights)
            
            #Check convergence
            delta = np.abs(self._lnp - prev_lnp)
            if delta < self.tol:
                print(f"Converged at iteration {iteration + 1}")
                break
        
        return self
    
    def _expectation(self):
        """E-step: Calculate responsibilities (posterior probabilities)."""
        n_samples = self.x.shape[0]
        
        #Calculate likelihood for each component
        likelihoods = np.zeros((n_samples, self.num_components))
        
        for k in range(self.num_components):
            cov_stable = self.cov[k] + np.eye(self.cov[k].shape[0]) * 1e-6
            likelihoods[:, k] = self._weights[k] * multivariate_normal.pdf(
                self.x, self.means[k], cov_stable
            )
        
        #Normalize to get responsibilities
        total_likelihood = np.sum(likelihoods, axis=1, keepdims=True)
        self._responsibilities = likelihoods / (total_likelihood + 1e-10)
    
    def _maximization(self):
        """M-step: Update parameters using current responsibilities."""
        n_samples, n_features = self.x.shape
        
        #Effective number of points assigned to each component
        N_k = np.sum(self._responsibilities, axis=0)
        
        #Update weights
        self._weights = N_k / n_samples
        
        #Update means
        self.means = np.zeros((self.num_components, n_features))
        for k in range(self.num_components):
            self.means[k] = np.sum(self._responsibilities[:, k:k+1] * self.x, axis=0) / (N_k[k] + 1e-10)
        
        #Update covariances
        self.cov = np.zeros((self.num_components, n_features, n_features))
        for k in range(self.num_components):
            diff = self.x - self.means[k]
            weighted_diff = self._responsibilities[:, k:k+1] * diff
            self.cov[k] = np.dot(weighted_diff.T, diff) / (N_k[k] + 1e-10)
            
            #Add small value to diagonal for numerical stability
            self.cov[k] += np.eye(n_features) * 1e-6
    
    def predict(self, x):
        """Predict cluster assignments for new data."""
        if self.means is None:
            raise ValueError("Model must be fitted before prediction")
        
        n_samples = x.shape[0]
        likelihoods = np.zeros((n_samples, self.num_components))
        
        for k in range(self.num_components):
            cov_stable = self.cov[k] + np.eye(self.cov[k].shape[0]) * 1e-6
            likelihoods[:, k] = self._weights[k] * multivariate_normal.pdf(
                x, self.means[k], cov_stable
            )
        
        #Return cluster with highest prob
        return np.argmax(likelihoods, axis=1)
    
    def predict_proba(self, x):
        """Predict probability of each cluster for new data."""
        if self.means is None:
            raise ValueError("Model must be fitted before prediction")
        
        n_samples = x.shape[0]
        likelihoods = np.zeros((n_samples, self.num_components))
        
        for k in range(self.num_components):
            cov_stable = self.cov[k] + np.eye(self.cov[k].shape[0]) * 1e-6
            likelihoods[:, k] = self._weights[k] * multivariate_normal.pdf(
                x, self.means[k], cov_stable
            )
        
        #Normalize to get prob
        total_likelihood = np.sum(likelihoods, axis=1, keepdims=True)
        return likelihoods / (total_likelihood + 1e-10)
    
    def score(self, x):
        """Calculate log-likelihood for given data."""
        return self._log_likelihood(x, self.means, self.cov, self._weights)



if __name__ == '__main__': 
    data = np.loadtxt("data/em_data.txt")
    x_train, x_test = train_test_split(data, test_size=0.4, train_size=0.6)
    model = GMM(2, 100)
    model.fit(x_train)
    model.predict(x_test)
    
    # Create plot 
    plt.title('Gaussian Mixture Model')
    plt.gcf().set_size_inches(8, 8)
    plt.xlim(np.min(data[:, 0]) - .5, np.max(data[:, 0]) + .5) 
    plt.ylim(np.min(data[:, 1]) - .5, np.max(data[:, 1]) + .5)
    
    # Get learned model parameters
    means, covs = model.means, model.cov 
    for k in range(2):
        # Visualize means
        plt.scatter(means[k, 0], means[k, 1], color='k', alpha=0.7, s=150, marker='X')
        # Visualize covariances as ellipses
        for fac in range(1, 4): 
            circle = plt.Circle(means[k], fac * np.sqrt(np.max(np.diag(covs[k]))), color='k', fill=False)
            plt.gca().add_artist(circle)
    
    # Visualize data points
    plt.scatter(data[:, 0], data[:, 1], alpha=.4) 
    plt.show()