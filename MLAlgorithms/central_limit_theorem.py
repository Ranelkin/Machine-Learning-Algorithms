
import numpy as np
import matplotlib.pyplot as plt


def sampling():
    # Draw N = 1000 numbers from standard uniform distribution.
    x = np.random.uniform(size=1000)
    # Compute histogram with 20 equal sized bins.
    hist, edges = np.histogram(x, bins=20, range=(0.0, 1.0)) # Normalize histogram.
    hist = hist / 1000
    # Plot the normalized histogram.
    plt.figure(figsize=(8, 5))
    # Set font sizes.
    plt.rc('axes', titlesize=16) 
    plt.rc('axes', labelsize=12) 
    plt.rc('xtick', labelsize=10) 
    plt.rc('ytick', labelsize=10)
    # Set title and label axes.
    plt.title('Sample $N = 1000$ times from $\mathcal{U}\,(0,1)$') 
    plt.xlabel('Unit interval divided into 20 bins') 
    plt.ylabel('Observations')
    # Show bar plot.
    plt.bar( x=edges[:-1],
    height=hist,
    # Compute width of the bars from edges.
    width=np.diff(edges), edgecolor="black",
    # Align the left edges of the bars with the x positions.
    align="edge" )
    plt.show()
    
def central_limit_theorem():
    # Draw N times K numbers from uniform distribution and compute means.
    batch_sizes = [2, 5, 10, 100]
    means = np.array([ np.mean(np.random.uniform(size=(1000, K)), axis=1) for K in batch_sizes
    ])
    # Show results as line plots.
    plt.figure(figsize=(8, 5)) 
    for k in range(len(means)):
        # Compute normalized histograms.
        hist, edges = np.histogram(means[k], bins=20, range=(0.0, 1.0)) 
        hist = hist / 1000
        # Make line plots for different values of K.
        plt.plot(
        0.5*(edges[1:] + edges[:-1]), hist ,
        label=f'$K = {batch_sizes[k]}$'
        )
        
    # Set title and label axes.
    plt.title('Distribution of $N = 1000$ Means of $K$ samples') 
    plt.xlabel('Unit interval divided into 20 bins') 
    plt.ylabel('Means')
    # Include legend and show plot.
    plt.legend() 
    plt.show()
        


if __name__ == '__main__': 
    central_limit_theorem()