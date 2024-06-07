import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt 

__author__ = "Ranel Karimov"

def k_means(x, k):
    #Data points
    num = len(x)
    #Store indices of clusters
    clusters = np.zeros(num)
    #Initialize random centroids
    cent_feat1 = np.random.choice([min([point[0] for point in x]), max([point[0] for point in x])], size=k)
    cent_feat2 = np.random.choice([min([point[1] for point in x]), max([point[1] for point in x])], size=k)
    centroids = list(zip(cent_feat1, cent_feat2))
    
    x = np.array(x)
    centroids = np.array(centroids)
    
    x = x[:, np.newaxis, :]
    centroids = centroids[np.newaxis, :,:]
   
    while True: 
        #Expectation Step 
        #Select the cluster with least squared distance 
        assignment = np.argmin(np.linalg.norm(x - centroids, axis=2), axis=1)
        
        #Check for convergence 
        if np.array_equal(clusters, assignment): 
            break
        clusters = assignment
        
        #Maximization Step 
        for i in range(centroids.shape[1]): 
            cluster_points = x[clusters==i, 0, :]
            centroids[0, i, :] = x[clusters == i].mean(axis=0) if np.any(clusters == i) else centroids[0, i, :]
   
    return clusters, centroids
        
    
    
if __name__ == '__main__': 
    #Task after implementation: Test the algorithm on the Iris dataset with num_clusters = 3. 
    # Visualize the results as 2D scatter plots, indicating clusters via color coding, 
    # for the following variables: sepal length versus sepal width and sepal length versus petal width.
    dataset = load_iris().data
    
    sepal_length = dataset[:, 0]
    sepal_width= dataset[:, 1]
    

    #Couldnt find better abbrevations 
    #Combine Features as stated in exercise 
    sepal_length_vs_width = list(zip(sepal_length, sepal_width))
   
    
    k = 3
    clusters, centroids = k_means(sepal_length_vs_width, k)
    
    #Plotting results 
    fig = plt.figure()
    ax = plt.axes()
    
    # Using cluster assignment to color the data points
    scatter = ax.scatter(sepal_length, sepal_width, c=clusters, cmap='viridis', marker='o', label='Data Points')
    

    # Plotting centroids
    ax.scatter([cent[0] for cent in centroids[0]], [cent[1] for cent in centroids[0]],
               c='red', marker='x', s=100, label='Centroids')

    ax.set_xlabel('Sepal Length')
    ax.set_ylabel('Sepal Width')
    ax.set_title('Cluster Analysis with Centroids')
    ax.legend()

    plt.show()