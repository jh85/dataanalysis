import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import umap
from sklearn.preprocessing import StandardScaler
import seaborn as sns

import scipy.stats as stats
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import multivariate_normal,chi2


def load_lympho(standardized=True, dim_reduced=True, seed=42):
    '''
    Returns Lymphography dataset
    https://odds.cs.stonybrook.edu/lympho/
    '''
    filename = "lympho.mat"
    d = sio.loadmat(filename)
    X = d["X"]
    y = d["y"].astype(int).reshape(-1)
    N = len(y)
    if standardized == True:
        X = StandardScaler().fit_transform(X)
    if dim_reduced == True:
        mapper = umap.UMAP(n_neighbors=N-1,n_jobs=1,random_state=seed)
        X = mapper.fit_transform(X)
    return X,y,N

def hotelling_t2(X, alpha=0.01):
    N = len(X)
    X_centered = X - np.mean(X,axis=0)
    covm = (X_centered.T @ X_centered) / (N-1)
    mahalanobis_distance = X_centered @ np.linalg.inv(covm) @ X_centered.T
    mahalanobis_distance = np.diag(mahalanobis_distance)
    dim = len(X[0])
    threshold = stats.chi2.ppf(q=1-alpha, df=dim, scale=1)
    return mahalanobis_distance > threshold

def knearestneighbors(X, n_neighbors):
    model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    model.fit(X)
    # Calculate the distances to the k nearest neighbors
    distances, indices = model.kneighbors(X)
    # Calculate the anomaly scores using average distance
    anomaly_scores = np.mean(distances, axis=1)
    threshold = 2.25
    return anomaly_scores > threshold

def gaussianmixture(X, n_components, probability):
    X = X - np.mean(X,axis=0)
    model = GaussianMixture(n_components=n_components, covariance_type="full")
    model.fit(X)
    d2s = [np.diag((X-m)@np.linalg.inv(s)@(X-m).T) for (m,s) in zip(model.means_,model.covariances_)]
    d2 = np.sum(np.array(d2s).T * model.predict_proba(X), axis=1)
    df = len(X[0])
    threshold = chi2.isf(probability,df)
    return d2 > threshold

def OCSVM(X):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma="auto")
    model.fit(X_scaled)
    scores = model.predict(X_scaled)
    return scores == -1

def main():
    seed = 1
    color = ["red","blue","yellow","green"]

    # Displaying original Lympho data with given labels
    X_orig,_,_ = load_lympho(standardized=False,dim_reduced=False)
    X_umap,y,N = load_lympho(standardized=True,dim_reduced=True,seed=seed)
    c = [color[y[i].astype(int)] for i in range(N)]
    plt.scatter(X_umap[:,0], X_umap[:,1], color=c)
    plt.title(f"Official Outliers: Red:data({N-6}) Blue:outlier(6)")
    plt.show()

    # Display histograms of all 18 features
    fig,ax = plt.subplots(6,3,figsize=(10,20))
    for c in range(len(X_orig[0])):
        row = c // 3
        col = c % 3
        his,bins = np.histogram(X_orig[:,c],bins=int(np.log2(N)))
        bins = (bins[:-1] + bins[1:]) / 2
        ax[row,col].bar(bins,his,align="edge", width=0.3)
        ax[row,col].set_title(f"Feature {c}")
    plt.show()

    # Elbow method to determine the best cluster size
    sses = dict()
    for k in range(2,21):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X_orig)
        sses[k] = kmeans.inertia_
    plt.plot(sses.keys(), sses.values(), marker="o")
    plt.xlim([2,20])
    plt.grid()
    plt.title(f"Optimal cluster number: 3")
    plt.show()
    

    # OC SVM
    label = OCSVM(X_orig)
    anomal_data = np.sum(label)
    normal_data = N - anomal_data
    c = [color[label[i].astype(int)] for i in range(N)]
    plt.scatter(X_umap[:,0], X_umap[:,1], color=c)
    plt.title(f"OCSVM: Red:data{normal_data} Blue:outlier{anomal_data}")
    plt.show()


    # GMM
    label = gaussianmixture(X_orig, n_components=3, probability=0.15)
    anomal_data = np.sum(label)
    normal_data = N - anomal_data
    c = [color[label[i].astype(int)] for i in range(N)]
    plt.scatter(X_umap[:,0], X_umap[:,1], color=c)
    plt.title(f"GMM: Red:data{normal_data} Blue:outlier{anomal_data}")
    plt.show()

    
    # k-Nearest Neighbor
    label = knearestneighbors(X_orig, n_neighbors=5)
    anomal_data = np.sum(label)
    normal_data = N - anomal_data
    c = [color[label[i].astype(int)] for i in range(N)]
    plt.scatter(X_umap[:,0], X_umap[:,1], color=c)
    plt.title(f"kNN: Red:data{normal_data} Blue:outlier{anomal_data}")
    plt.show()

    
    # Hotelling test
    label = hotelling_t2(X_orig, 0.005)
    anomal_data = np.sum(label)
    normal_data = N - anomal_data
    c = [color[label[i].astype(int)] for i in range(N)]
    plt.scatter(X_umap[:,0], X_umap[:,1], color=c)
    plt.title(f"Hosteling T2: Red:data{normal_data} Blue:outlier{anomal_data}")
    plt.show()

main()
