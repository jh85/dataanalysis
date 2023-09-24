import numpy as np
from sklearn.datasets import make_blobs
from scipy.stats import mode
from collections import Counter
import time

def is_tree(obj):
    return type(obj) == dict
    
def predict_tree(tree, x):
    feature_idx = tree["feature_idx"]
    split_point = tree["split_point"]
    
    if x[feature_idx] <= split_point:
        if is_tree(tree["left"]):
            return predict_tree(tree["left"], x)
        else:
            return tree["left"]
    else:
        if is_tree(tree["right"]):
            return predict_tree(tree["right"], x)
        else:
            return tree["right"]

def predict(trees, X):
    preds = []
    for i in range(len(X)):
        p = mode([predict_tree(t,X[i]) for t in trees]).mode
        preds.append(p)
    return np.array(preds)

def draw_bootstrap(X,y):
    n_samp = len(X)
    bootstrap_indices = np.random.choice(range(n_samp), n_samp, replace=True)
    oob_indices = [i for i in range(n_samp) if i not in bootstrap_indices]
    X_bootstrap = X[bootstrap_indices]
    y_bootstrap = y[bootstrap_indices]
    X_oob = X[oob_indices]
    y_oob = y[oob_indices]
    return X_bootstrap, y_bootstrap, X_oob, y_oob

def random_forest(X,y,n_estimators,max_depth,min_length,max_features):
    trees = []
    oobs = []
    for i in range(n_estimators):
        X_bootstrap,y_bootstrap,X_oob,y_oob = draw_bootstrap(X,y)
        t = build_tree(X_bootstrap, y_bootstrap,max_depth,min_length,max_features)
        trees.append(t)
    return trees

def build_tree(X,y,max_depth,min_length,max_features):
    root = split_data(X,y,max_features)
    split_node(root, max_features, min_length, max_depth, depth=1)
    return root

def entropy(arr):
    total_count = len(arr)
    if total_count == 0:
        return 0
    total = 0
    freq = Counter(arr)
    for k,count in freq.items():
        p = count / total_count
        total += -p*np.log2(p)
    return total

def calc_info_gain(l,r):
    p = l + r
    return entropy(p) - len(l)/len(p) * entropy(l) - len(r)/len(p) * entropy(r)

def gini_index(arr):
    total_count = len(arr)
    if total_count == 0:
        return 0
    total = 0
    freq = Counter(arr)
    for k,count in freq.items():
        p = count / total_count
        total += p*p
    return 1 - total

def calc_gini_impurity(l,r):
    p = l + r
    return gini_index(p) - len(l)/len(p) * gini_index(l) - len(r)/len(p) * gini_index(r)
    
def split_data(X,y,max_features):
    n_features = len(X[0])
    features = np.random.choice(range(n_features),max_features,replace=False)
    best_info_gain = -float("inf")
    node = None
    for feature_idx in features:
        arr =  X[:,feature_idx]
        for split_point in np.linspace(np.min(arr), np.max(arr), 100):
            left = {"X_bootstrap":[], "y_bootstrap":[]}
            right = {"X_bootstrap":[], "y_bootstrap":[]}
            for i,value in enumerate(X[:,feature_idx]):
                    if value <= split_point:
                        left["X_bootstrap"].append(X[i])
                        left["y_bootstrap"].append(y[i])
                    else:
                        right["X_bootstrap"].append(X[i])
                        right["y_bootstrap"].append(y[i])
            #split_info_gain = calc_info_gain(left["y_bootstrap"], right["y_bootstrap"])
            split_info_gain = calc_gini_impurity(left["y_bootstrap"], right["y_bootstrap"])
            if split_info_gain > best_info_gain:
                best_info_gain = split_info_gain
                left["X_bootstrap"] = np.array(left["X_bootstrap"])
                left["y_bootstrap"] = np.array(left["y_bootstrap"])
                right["X_bootstrap"] = np.array(right["X_bootstrap"])
                right["y_bootstrap"] = np.array(right["y_bootstrap"])

                node = {"info_gain": split_info_gain,
                        "left": left,
                        "right": right,
                        "feature_idx": feature_idx,
                        "split_point": split_point}
    return node

def split_node(node, max_features, min_length, max_depth, depth):
    left = node["left"]
    right = node["right"]
    del(node["left"])
    del(node["right"])
    if len(left["y_bootstrap"]) == 0 or len(right["y_bootstrap"]) == 0:
        ymode = mode(left["y_bootstrap"] + right["y_bootstrap"]).mode
        node["left"] = ymode
        node["right"] = ymode
    elif depth >= max_depth:
        node["left"] = mode(left["y_bootstrap"]).mode
        node["right"] = mode(right["y_bootstrap"]).mode
    else:
        if len(left["y_bootstrap"]) <= min_length:
            ymode = mode(left["y_bootstrap"]).mode
            node["left"] = ymode
            node["right"] = ymode
        else:
            node["left"] = split_data(left["X_bootstrap"], left["y_bootstrap"], max_features)
            split_node(node["left"], max_features, min_length, max_depth, depth+1)
        if len(right["y_bootstrap"]) <= min_length:
            ymode = mode(right["y_bootstrap"]).mode
            node["left"] = ymode
            node["right"] = ymode
        else:
            node["right"] = split_data(right["X_bootstrap"], right["y_bootstrap"], max_features)
            split_node(node["right"], max_features, min_length, max_depth, depth+1)

def gen_data(n_samples, n_features, n_categories, is_random=False):
    n_range = 10
    if is_random:
        X = np.random.choice(range(n_range), n_samples * n_features).reshape(n_samples, n_features)
        y = np.random.choice(range(n_categories), n_samples)
        return X,y
    else:
        X,y = make_blobs(n_samples=n_samples, centers=n_categories, n_features=n_features, random_state=42)
        f = np.vectorize(lambda x: round(x*2))
        X = f(X)
        return X,y
            
def main():
    n_samples = 10000
    n_features = 5
    n_range = 10
    n_categories = 4

    X,y = gen_data(n_samples, n_features, n_categories, is_random=False)
    model = random_forest(X, y, n_estimators=4, max_depth=3, min_length=10, max_features=n_features)
    y2 = predict(model, X)
    print("accuracy:", np.sum(y == y2) / n_samples)

    X,y = gen_data(n_samples, n_features, n_categories, is_random=True)
    y2 = predict(model, X)
    print("accuracy:", np.sum(y == y2) / n_samples)

    return

main()
