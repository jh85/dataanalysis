import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import math
import time
import datetime
import seaborn as sns

# To use cuml, enable this
is_gpu_available = False

if is_gpu_available:
    from cuml.cluster import KMeans
    from cuml.metrics.cluster import silhouette_samples,silhouette_score
    from cuml.preprocessing import RobustScaler
    from cuml.decomposition import PCA
    from cuml.manifold.umap import UMAP
else:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples,silhouette_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import umap


# Title: New York City Civil Service List
# URL: https://data.cityofnewyork.us/City-Government/Civil-Service-List-Active-/vx8i-nprf
# File: Civil_Service_List__Active_.csv

def umap_embed(df, dim=2, intersection=True):
    numerical = df.select_dtypes(include="float")
    for c in numerical.columns:
        numerical[c] = (numerical[c] - numerical[c].mean())/numerical[c].std(ddof=0)
    fit1 = umap.UMAP(n_components=dim).fit(numerical)

    categorical = df.select_dtypes(include="object")
    if categorical.empty:
        return fit1.embedding_
    
    categorical = pd.get_dummies(categorical)
    fit2 = umap.UMAP(metric="dice", n_neighbors=2500, n_components=dim).fit(categorical)
    # intersection will resemble the numerical embedding more.
    # union will resemble the categorical embedding more.
    if intersection:
        embedding = fit1 * fit2
    else:
        embedding = fit1 + fit2
    umap_embedding = embedding.embedding_
    return umap_embedding

def sort_freq(freq,key):
    # when sorting by key(0), use ascending order
    # when sorting by value(1), use descending order
    is_reverse = False if key==0 else True
    return {k:v for k,v in sorted(freq.items(), key=lambda itm:itm[key], reverse=is_reverse)}

def show_frequency(df, column_name, top_n=100, sortby_x=False, drop_other=False, logscale=False,
                   rotation=0, filename=None, title=None, xlabel=None, ylabel=None):
    arr = df[column_name].tolist()
    freq = defaultdict(int)
    for i in range(len(arr)):
        freq[str(arr[i])] += 1

    freq_sorted = None
    if sortby_x:
        freq_sorted = sort_freq(freq,0)
    else:
        freq_sorted = sort_freq(freq,1)
       
    x,y = None,None
    if len(freq) > top_n:
        if drop_other:
            x = list(freq_sorted.keys())[:top_n]
            y = list(freq_sorted.values())[:top_n]
        else:
            x = list(freq_sorted.keys())[:top_n] + ["other"]
            y_last = sum(list(freq_sorted.values())[top_n:])
            y = list(freq_sorted.values())[:top_n] + [y_last]
    else:
        x = list(freq_sorted.keys())
        y = list(freq_sorted.values())

    plt.bar(x,y)
    if xlabel:
        plt.xlabel(xlabel)
    else:
        plt.xlabel(column_name)
    if ylabel:
        plt.ylabel(ylabel)
    else:
        plt.ylabel("Frequency")
    if title:
        plt.title(title)
    else:
        plt.title(f"Frequency of {column_name}")
    if logscale:
        plt.yscale("log")
    if sortby_x:
        xticks = x[0:len(x):len(x)//10]
        plt.xticks(xticks, xticks, rotation=rotation, ha="right")
    else:
        plt.xticks(rotation=rotation, ha="right")
    plt.grid()
    plt.show()
    plt.clf()

def show_histogram(df, column_name, logscale=False, logscale_x=False, remove_na=False,
                   rotation=0, filename=None, title=None, xlabel=None, ylabel=None,
                   is_date=False):
    arr = df[column_name].tolist()
    if remove_na:
        arr = df[column_name].dropna().tolist()
   
    if logscale_x:
        for i in range(len(arr)):
            if arr[i] < 1:
                arr[i] = 1
            arr[i] = np.log10(arr[i])
    max_val = max(arr)
    min_val = min(arr)
    bin_num = 1 + int(np.log2(len(arr))) # Sturges rule
    bin_width = (max_val - min_val) / bin_num
    bins = np.linspace(min_val, max_val, bin_num+1)
    # increase the last bin slightly so that it is greater than max value
    bins[-1] += 1
    freq = defaultdict(int)
    for v in arr:
        for i in range(len(bins)-1):
            if bins[i] <= v < bins[i+1]:
                freq[bins[i]] += 1
                break
    freq_sorted = sort_freq(freq,0)
    x = list(freq_sorted.keys())
    y = list(freq_sorted.values())

    plt.bar(x,y,width=bin_width*0.9)
    if xlabel:
        plt.xlabel(xlabel)
    else:
        plt.xlabel(column_name)
    if ylabel:
        plt.ylabel(ylabel)
    else:
        plt.ylabel("Frequency")
    if title:
        plt.title(title)
    else:
        plt.title(f"Frequency of {column_name}")
    if logscale:
        plt.yscale("log")
    if logscale_x:
        x10 = list(map(lambda v:round(v), 10**np.array(x)))
        x10[0] = 0
        xticks = []
        for i in range(len(x10)-1):
            xticks.append(f"{x10[i]}-{x10[i+1]}")
        xticks.apppend(f"{x10[-1]}-")
        plt.xticks(x, xticks, rotation=rotation, ha="right")
    else:
        xticks = list(map(lambda t:round(t), x))
        plt.xticks(x, xticks, rotation=rotation, ha="right")
    plt.grid()
    plt.show()
    plt.clf()

def show_distribution(n_clusters, pred, desc):
    freq = {str(k):0 for k in range(n_clusters)}
    for c in pred:
        freq[str(c)] += 1
    freq2 = sort_freq(freq,1)
    x = list(freq2.keys())
    y = list(freq2.values())
    plt.bar(x,y)
    plt.title(f"Size Distribution of Candidate Group clustering={n_clusters}")
    plt.xlabel("Candidate Group")
    plt.ylabel("Number of Candidates")
    plt.show()
    plt.clf()

def show_silhouette(df, n_clusters, pred, silhouettes, score, desc):
    plt.xlim([-1, 1])
    plt.ylim([0, len(df) + (n_clusters + 1) * 10])
    y_lower = 10
    for i in range(n_clusters):
        silhouette_ith = silhouettes.get()[pred == i]
        silhouette_ith.sort()
        size_ith = len(silhouette_ith)
        y_upper = y_lower + size_ith
        plt.fill_betweenx(np.arange(y_lower, y_upper),0,silhouette_ith)
        y_lower = y_upper + 10
    plt.axvline(x=score, color="red", linestyle="--")
    plt.title(f"Silhouette Plot for candidate groups clustering={n_clusters}")
    plt.xlabel("Silhouette Score values")
    plt.ylabel("Candidate Group labels")
    plt.show()
    plt.clf()

def convert_to_timestamp(x):
    return time.mktime(pd.to_datetime(x).timetuple())

def run_kmeans(df, cluster_sizes, desc, return_preds=False):
    colors = ["blue","red","green","purple","orange",  "brown",
              "pink","gray","olive","cyan", "magenta", "gold",
              "yellowgreen", "navy", "lightsteelblue", "teal", "peru"]
    sses = dict()
    sils = dict()
    for k in cluster_sizes:
        model = KMeans(n_clusters=k, n_init=10)
        preds = model.fit_predict(df)
        sses[k] = model.inertia_
        preds = renumbering(preds, k)

        if return_preds:
            return preds
        
        #
        # mapping KMeans clustering onto result
        #
        plt.scatter(df.iloc[:,0], df.iloc[:,1], c=list(map(lambda cl:colors[cl], preds)))
        plt.title(f"KMeans clustering result on {desc} map. cluster number: {k}")
        plt.xlabel(f"{desc} Component 1")
        plt.ylabel(f"{desc} Component 2")
        recs = []
        for i in range(k):
            recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[i]))
        plt.legend(recs,list(np.arange(k)))
        plt.grid()
        plt.show()
        plt.clf()

        #
        # Silhouette Chart
        #
        sil_scores = silhouette_samples(df.values, preds)
        avg_score = float(np.mean(sil_scores))
        sils[k] = avg_score
        show_silhouette(df, k, preds, sil_scores, avg_score, desc)
        show_distribution(k, preds, desc)

    #
    # Elbow method with KMeans SSE
    #
    plt.scatter(sses.keys(), sses.values())
    plt.title("Elbow method result: Number of Clusters vs SSE by {desc}")
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.grid()
    plt.show()
    plt.clf()

    #
    # Elbow method with Silhouette score
    #
    plt.scatter(sils.keys(), sils.values())
    plt.title(f"Elbow method result: Number of Clusters vs Silhouette score by {desc}")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.grid()
    plt.show()
    plt.clf()

def renumbering(arr,n_clusters):
    '''
    reassign cluster numbers in the K-Means prediction (arr)
    so that they are in descending order of frequency, starting with 0, 1, 2, etc.,
    '''
    freq = {k:0 for k in range(n_clusters)}
    for v in arr:
        freq[v] += 1
    freq2 = sort_freq(freq,1)
    renum_rule = {k:i for i,k in enumerate(freq2.keys())}
    for i in range(len(arr)):
        arr[i] = renum_rule[arr[i]]
    return arr

def pca(df):
    '''
    1. prepare data
    2. standardize data  // RobustScaler()
    3. one-hot encoding  // get_dummies()
    4. PCA               // cuml.decomposition.PCA
    5. K-Means           // cuml.cluster.KMeans
    '''
    dropped = ["First Name", "MI", "Last Name", "List Title Desc",
               "List Agency Desc", "List Div Code"]
    df = df.drop(columns=dropped)

    rpl_vc = {"":0.0,"Veteran's Credit":     5.0,"Disabled Veteran's Credit":10.0}
    rpl_pc = {"":0.0,"Parent Legacy Credit": 10.0}
    rpl_sc = {"":0.0,"Sibling Legacy Credit":10.0}
    rpl_rc = {"":0.0,"Residency Credit":     5.0}
    df["Veteran Credit"]     = df["Veteran Credit"].replace(rpl_vc)
    df["Parent Lgy Credit"]  = df["Parent Lgy Credit"].replace(rpl_pc)
    df["Sibling Lgy Credit"] = df["Sibling Lgy Credit"].replace(rpl_sc)
    df["Residency Credit"]   = df["Residency Credit"].replace(rpl_rc)

    date_columns = ["Published Date", "Established Date",
                    "Anniversary Date", "Extension Date"]
    days_added = 31
    for c in date_columns:
        df[c] = pd.to_datetime(df[c], format="%m/%d/%Y", errors="coerce").dt.date
        non_empty = list(filter(lambda day: pd.isnull(day)==False, df[c].tolist()))
        empty_value = max(non_empty) + datetime.timedelta(days_added)
        df[c] = df[c].replace({pd.NaT: empty_value})
        df[c] = df[c].apply(convert_to_timestamp)

    # save the dataframe before one-hot encoding for later use
    df_orig = df.copy()
    df = pd.get_dummies(df)
    
    transformer = RobustScaler().fit(df.values)
    df = pd.DataFrame(transformer.transform(df.values), columns=df.columns)

    pca = PCA(n_components=2)
    result = pca.fit_transform(df)

    #
    # Display Principal Component 1
    #
    pc1_loc = 0
    top_n = 20
    y = pca.components_.iloc[pc1_loc,:].tolist()
    x = np.arange(len(y))
    freq = {str(k):v for k,v in zip(x,y)}
    freq2 = sort_freq(freq,1)
    x = list(freq2.keys())[:top_n]
    y = list(freq2.values())[:top_n]
    plt.bar(x,y)
    plt.grid()
    plt.title(f"{top_n} most correlated columns to Principal component 1")
    plt.xlabel("Column name (One Hot representation)")
    plt.ylabel("Factor score to Principal Component 1")
    plt.xticks(x, list(map(lambda s:df.columns[int(s)],x)), rotation=45, ha="right")
    plt.show()
    plt.clf()

    #
    # Display Principal Component 2
    #
    pc2_loc = 1
    top_n = 20
    y = pca.components_.iloc[pc2_loc,:].tolist()
    x = np.arange(len(y))
    freq = {str(k):v for k,v in zip(x,y)}
    freq2 = sort_freq(freq,1)
    x = list(freq2.keys())[:top_n]
    y = list(freq2.values())[:top_n]
    plt.bar(x,y)
    plt.grid()
    plt.title(f"{top_n} most correlated columns to Principal component 2")
    plt.xlabel("Column name (One Hot representation)")
    plt.ylabel("Factor score to Principal Component 2")
    plt.xticks(x, list(map(lambda s:df.columns[int(s)],x)), rotation=45, ha="right")
    plt.show()
    plt.clf()

    # Now run K-Means on the result of PCA
    run_kmeans(result, list(range(2,16)), "pca")

    result_umap = None
    if is_gpu_available:
        # Then run K-Means on the result of UMAP
        # https://umap-learn.readthedocs.io/en/latest/parameters.html
        # n_neighbors:
        # This parameter controls how UMAP balances local versus global structure in the data.
        # Low values of n_neighbors will force UMAP to concentrate on very local structure.
        # Large values will push UMAP to look at larger neighborhoods, losing fine detail structure
        # for the sake of getting the broader of the data.
        # 300 worked in my environment but 500 failed due to out of memory.
        result_umap = UMAP(n_neighbors=300, n_components=2, init="spectral").fit_transform(df)
    else:
        result_umap = umap_embed(df_orig, dim=2)
    run_kmeans(result_umap, list(range(2,16)), "umap")

    preds = run_kmeans(result, [4], "pca", return_preds=True)
    df_orig["cluster"] = preds

    # convert categorical columns to integer to calculate the means
    for c in ["Exam No", "List Title Code", "List Agency Code"]:
        df_orig[c] = df_orig[c].astype(int)
    
    for k in range(4):
        print(f"class-{k} size: ", len(df_orig[df_orig["cluster"]==k]))
        print(f"class-{k} means\n", df_orig[df_orig["cluster"]==k].mean())


def main():
    data_file = "Civil_Service_List__Active_.csv"
    df = pd.read_csv(data_file, low_memory=False)

    columns_nan = ["Published Date","Established Date","Anniversary Date","Extension Date",
                   "Veteran Credit","Parent Lgy Credit","Sibling Lgy Credit","Residency Credit"]
    for c in columns_nan:
        df[c] = df[c].replace(np.nan, "")
    categorical = ["Exam No", "List Title Code", "List Agency Code"]
    for c in categorical:
        df[c] = df[c].astype(str)
    df["Group No"] = df["Group No"].astype(float)

    pca(df)

main()

