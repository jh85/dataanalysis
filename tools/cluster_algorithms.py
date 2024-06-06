import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
from collections import deque, Counter
import scipy.stats

def circle_data(r, N):
    x = []
    y = []
    for i in range(N):
        t = np.random.uniform(low=-np.pi, high=np.pi, size=None)
        d1 = r*np.cos(t) + np.random.normal(loc=0,scale=1,size=None)
        d2 = r*np.sin(t) + np.random.normal(loc=0,scale=1,size=None)
        x.append(d1)
        y.append(d2)
    return x,y

def toy_data_example1():
    N = 50
    x1,y1 = circle_data(10,N)
    x2,y2 = circle_data(20,N)
    df = pd.DataFrame()
    df["x"] = x1 + x2
    df["y"] = y1 + y2

    fig,ax = plt.subplots(4,2, figsize=(10,20))
    colors = ["red","blue","green","tan","pink"] + ["yellow" for _ in range(50)]

    # K-Means
    km = KMeans(n_clusters=2, n_init="auto")
    preds = km.fit_predict(df)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[0,0].scatter(d[i,0], d[i,1], color=c[i])
    ax[0,0].set_title("sklean K-Means")
    
    preds = myKMeans(df,2)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[0,1].scatter(d[i,0], d[i,1], color=c[i])
    ax[0,1].set_title("My K-Means")
    
    # DBSCAN N:1000 eps:2
    ds = DBSCAN(eps=6.5, min_samples=5)
    preds = ds.fit_predict(df)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[1,0].scatter(d[i,0], d[i,1], color=c[i])
    ax[1,0].set_title("sklearn DBSCAN")
    
    preds = myDBSCAN(df,eps=6.5)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[1,1].scatter(d[i,0], d[i,1], color=c[i])
    ax[1,1].set_title("My DBSCAN")

    # Gaussian Mixture
    gm = GaussianMixture(n_components=2)
    preds = gm.fit_predict(df)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[2,0].scatter(d[i,0], d[i,1], color=c[i])
    ax[2,0].set_title("sklearn Gaussian Mixture")

    preds = myGaussianMixture(df,n_clusters=2)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[2,1].scatter(d[i,0], d[i,1], color=c[i])
    ax[2,1].set_title("My Gaussian Mixture")

    # Agglomerative Clustering
    ag = AgglomerativeClustering(n_clusters=2)
    preds = ag.fit_predict(df)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[3,0].scatter(d[i,0], d[i,1], color=c[i])
    ax[3,0].set_title("sklearn Agglomerative Clustering")

    preds = myAgglomerativeClustering(df,n_components=2)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[3,1].scatter(d[i,0], d[i,1], color=c[i])
    ax[3,1].set_title("My Agglomerative Clustering")

    plt.show()

def toy_data_example2():
    mu1 = np.array([-10,10])
    var1 = np.diag([3,5])
    g1 = np.random.multivariate_normal(mean=mu1, cov=var1, size=300)

    mu2 = np.array([10,0])
    var2 = np.diag([3,2])
    g2 = np.random.multivariate_normal(mean=mu2, cov=var2, size=300)

    mu3 = np.array([0,-10])
    var3 = np.diag([3,2])
    g3 = np.random.multivariate_normal(mean=mu3, cov=var3, size=400)

    g = np.vstack([g1,g2,g3])
    df = pd.DataFrame()
    df["x"] = g[:,0]
    df["y"] = g[:,1]

    fig,ax = plt.subplots(3,2, figsize=(10,10))

    colors = ["red","blue","green","tan","pink","yellow"]

    # K-Means
    km = KMeans(n_clusters=3, n_init="auto")
    preds = km.fit_predict(df)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[0,0].scatter(d[i,0], d[i,1], color=c[i])
    ax[0,0].set_title("sklean K-Means")
    
    preds = myKMeans(df,3)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[0,1].scatter(d[i,0], d[i,1], color=c[i])
    ax[0,1].set_title("My K-Means")

    
    # DBSCAN
    ds = DBSCAN(eps=2, min_samples=5)
    preds = ds.fit_predict(df)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[1,0].scatter(d[i,0], d[i,1], color=c[i])
    ax[1,0].set_title("sklearn DBSCAN")
    
    preds = myDBSCAN(df,eps=2)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[1,1].scatter(d[i,0], d[i,1], color=c[i])
    ax[1,1].set_title("My DBSCAN")

    
    # Gaussian Mixture
    gm = GaussianMixture(n_components=3)
    preds = gm.fit_predict(df)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[2,0].scatter(d[i,0], d[i,1], color=c[i])
    ax[2,0].set_title("sklearn Gaussian Mixture")

    preds = myGaussianMixture(df,n_clusters=3)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[2,1].scatter(d[i,0], d[i,1], color=c[i])
    ax[2,1].set_title("My Gaussian Mixture")

    plt.show()

# KMeans implementation
def distance(x,y):
    return np.linalg.norm(x-y)

def myKMeans(data, K, T=100):
    N = len(data)
    label = np.random.choice(range(K), size=N)
    centroid = dict()

    for _ in range(T):
        for i in range(K):
            centroid[i] = data[label == i].mean().to_numpy()
        for i in range(N):
            min_d = 1e9
            min_c = -1
            for j in range(K):
                d = distance(data.iloc[i].to_numpy(), centroid[j])
                if d < min_d:
                    min_d = d
                    min_c = j
            label[i] = min_c
    return label

# DBSCAN implementation
def neighbors(u,df,eps):
    N = len(df)
    ns = []
    for v in range(N):
        if v == u:
            continue
        if distance(df.iloc[v].to_numpy(), df.iloc[u].to_numpy()) <= eps:
            ns.append(v)
    return ns

def myDBSCAN(data, eps):
    N = len(data)
    clusters = {i:-1 for i in range(N)}
    K = 0
    for u in range(N):
        if clusters[u] == -1:
            clusters[u] = K
            K += 1
            q = deque([u])
            while q:
                v = q.popleft()
                for nv in neighbors(v,data,eps):
                    if clusters[nv] == -1:
                        clusters[nv] = clusters[v]
                        q.append(nv)
    return np.array(list(clusters.values()))

# Gaussian Mixture implementation
def myGaussianMixture(df,n_clusters,T=200):
    N = len(df)
    n_features = len(df.columns)

    mu = np.random.normal(size=[n_clusters, n_features])
    sigma = np.array([np.eye(n_features)]*n_clusters)
    pi = np.array([1/n_clusters for _ in range(n_clusters)])
    gamma = np.zeros([N, n_clusters])
    log_likelihood = -np.inf

    data = df.values
    for _ in range(T):
        # E Step
        for k in range(n_clusters):
            pdf = scipy.stats.multivariate_normal(mean=mu[k], cov=sigma[k], allow_singular=True).pdf(data)
            gamma[:,k] = pi[k] * pdf
        gamma /= gamma.sum(axis=1, keepdims=True)

        # M step
        N_k = gamma.sum(axis=0)
        mu = gamma.T @ data / N_k[:, np.newaxis]
        for k in range(n_clusters):
            diff = data - mu[k]
            sigma[k] = (gamma[:, k, np.newaxis, np.newaxis] * diff[:, :, np.newaxis] @ diff[:, np.newaxis, :]).sum(axis=0) / N_k[k]
        pi = N_k / N

        # calculate log likelihood
        log_likelihood_new = np.log(gamma.sum(axis=1)).sum()
        if np.abs(log_likelihood_new - log_likelihood) < 1e-10:
            break
        log_likelihood = log_likelihood_new

    preds = gamma.argmax(axis=1)
    return preds

# Agglomerative Clustering implementation
def cluster_argmin(distance_matrix):
    minx,miny = (0,0)
    min_val = 1e10
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            if i == j:
                continue
            else:
                if distance_matrix[i,j] < min_val:
                    min_val = distance_matrix[i,j]
                    minx = i
                    miny = j
    return minx,miny

def cluster_distance(df,cluster_members):
    n_clusters = len(cluster_members)
    keys = list(cluster_members.keys())
    distance_matrix = np.zeros([n_clusters, n_clusters])
    for i in range(n_clusters):
        ith_elems = cluster_members[keys[i]]
        for j in range(i,n_clusters):
            jth_elems = cluster_members[keys[j]]
            
            d_in_clusters = euclidean_distances(df.iloc[ith_elems].to_numpy(),
                                                df.iloc[jth_elems].to_numpy())
            dij = np.max(d_in_clusters) # complete linkage
            # dij = np.min(d_in_clusters) # single linkage
            distance_matrix[i,j] = dij
            distance_matrix[j,i] = dij
    return distance_matrix

def myAgglomerativeClustering(data,n_components):
    N = len(data)
    cluster_members = {i:[i] for i in range(N)}
    Z = np.zeros([N-1,4])
    for i in range(N-1):
        if len(cluster_members) <= n_components:
            break
        cnames = list(cluster_members.keys())
        D = cluster_distance(data,cluster_members)
        x,y = cluster_argmin(D)
        cx = cnames[x]
        cy = cnames[y]
        cluster_members[i+N] = cluster_members[cx]+cluster_members[cy]
        Z[i,:] = cnames[x],cnames[y],D[x,y],len(cluster_members[i+N])
        del cluster_members[cx]
        del cluster_members[cy]
    labels = np.zeros(N, dtype=int)
    for i,vs in enumerate(cluster_members.values()):
        labels[vs] = i
    return labels

def umap_clustering_example():
    filename = "aliraza-indian-premier-league-2018-batting-and-bowling-data/data/total_data_na.csv"
    df = pd.read_csv(filename)
    
    for c in ["avg_x","avg_y","sr_y"]:
        mean_val = df[c].mean()
        df.fillna({c: mean_val}, inplace=True)

    data_cols = df.columns.tolist()
    data_cols.remove("player")
    df_data = df[data_cols]

    scaled_data = StandardScaler().fit_transform(df_data.values)
    df_data = pd.DataFrame(scaled_data, columns=data_cols)

    mapper = umap.UMAP(n_neighbors=70)
    d = mapper.fit_transform(df_data)
    df = pd.DataFrame(data=d, columns=["x","y"])

    plt.scatter(df.x, df.y)
    plt.show()

    fig,ax = plt.subplots(4,2, figsize=(10,20))

    colors = ["red","blue","green","tan","pink"] + ["yellow" for _ in range(50)]
    K = 3

    # K-Means
    km = KMeans(n_clusters=K, n_init="auto")
    preds = km.fit_predict(df)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[0,0].scatter(d[i,0], d[i,1], color=c[i])
    ax[0,0].set_title("sklean K-Means")
    
    preds = myKMeans(df,K)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[0,1].scatter(d[i,0], d[i,1], color=c[i])
    ax[0,1].set_title("My K-Means")

    
    # DBSCAN
    ds = DBSCAN(eps=0.7, min_samples=5)
    preds = ds.fit_predict(df)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[1,0].scatter(d[i,0], d[i,1], color=c[i])
    ax[1,0].set_title("sklearn DBSCAN")
    
    preds = myDBSCAN(df,eps=0.7)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[1,1].scatter(d[i,0], d[i,1], color=c[i])
    ax[1,1].set_title("My DBSCAN")

    
    # Gaussian Mixture
    gm = GaussianMixture(n_components=K)
    preds = gm.fit_predict(df)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[2,0].scatter(d[i,0], d[i,1], color=c[i])
    ax[2,0].set_title("sklearn Gaussian Mixture")

    preds = myGaussianMixture(df,n_clusters=K)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[2,1].scatter(d[i,0], d[i,1], color=c[i])
    ax[2,1].set_title("My Gaussian Mixture")

    # Agglomerative Clustering
    ag = AgglomerativeClustering(n_clusters=K)
    preds = ag.fit_predict(df)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[3,0].scatter(d[i,0], d[i,1], color=c[i])
    ax[3,0].set_title("sklearn Agglomerative Clustering")

    preds = myAgglomerativeClustering(df,n_components=K)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[3,1].scatter(d[i,0], d[i,1], color=c[i])
    ax[3,1].set_title("My Agglomerative Clustering")

    plt.show()


def umap_clustering_example2():
    filename = "aliraza-indian-premier-league-2018-batting-and-bowling-data/data/total_data_na.csv"
    df = pd.read_csv(filename)
    
    for c in ["avg_x","avg_y","sr_y"]:
        mean_val = df[c].mean()
        df.fillna({c: mean_val}, inplace=True)

    data_cols = df.columns.tolist()
    data_cols.remove("player")
    df_data = df[data_cols]

    scaled_data = StandardScaler().fit_transform(df_data.values)
    df_data = pd.DataFrame(scaled_data, columns=data_cols)

    mapper = umap.UMAP(n_neighbors=142)
    d = mapper.fit_transform(df_data)
    df = pd.DataFrame(data=d, columns=["x","y"])

    plt.scatter(df.x, df.y)
    plt.show()

    fig,ax = plt.subplots(4,2, figsize=(10,20))

    colors = ["red","blue","green","tan","pink"] + ["yellow" for _ in range(200)]
    K = 2

    # K-Means
    km = KMeans(n_clusters=K, n_init="auto")
    preds = km.fit_predict(df_data)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[0,0].scatter(d[i,0], d[i,1], color=c[i])
    ax[0,0].set_title("sklean K-Means")
    
    preds = myKMeans(df_data,K,T=200)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[0,1].scatter(d[i,0], d[i,1], color=c[i])
    ax[0,1].set_title("My K-Means")

    
    # DBSCAN
    ds = DBSCAN(eps=2.5, min_samples=5)
    preds = ds.fit_predict(df_data)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[1,0].scatter(d[i,0], d[i,1], color=c[i])
    ax[1,0].set_title("sklearn DBSCAN")
    
    preds = myDBSCAN(df_data,eps=2.5)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[1,1].scatter(d[i,0], d[i,1], color=c[i])
    ax[1,1].set_title("My DBSCAN")

    
    # Gaussian Mixture
    gm = GaussianMixture(n_components=K)
    preds = gm.fit_predict(df_data)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[2,0].scatter(d[i,0], d[i,1], color=c[i])
    ax[2,0].set_title("sklearn Gaussian Mixture")

    preds = myGaussianMixture(df_data,n_clusters=K,T=400)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[2,1].scatter(d[i,0], d[i,1], color=c[i])
    ax[2,1].set_title("My Gaussian Mixture")

    # Agglomerative Clustering
    ag = AgglomerativeClustering(n_clusters=K,linkage="ward")
    preds = ag.fit_predict(df_data)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[3,0].scatter(d[i,0], d[i,1], color=c[i])
    ax[3,0].set_title("sklearn Agglomerative Clustering")

    preds = myAgglomerativeClustering(df_data,n_components=K)
    c = [colors[preds[i]] for i in range(len(df))]
    d = df.values
    for i in range(len(df)):
        ax[3,1].scatter(d[i,0], d[i,1], color=c[i])
    ax[3,1].set_title("My Agglomerative Clustering")

    plt.show()

def main():
    toy_data_example1()
    toy_data_example2()
    umap_clustering_example()
    umap_clustering_example2()

main()
