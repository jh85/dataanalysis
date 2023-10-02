import numpy as np
from sklearn.datasets import make_blobs

def predict_tree(tree, x):
    if tree["left"] is None:
        A = tree["A"]
        B = tree["B"]
        return x@A + B
    else:
        feature_idx = tree["feature_idx"]
        split_point = tree["split_point"]
        if x[feature_idx] <= split_point:
            return predict_tree(tree["left"], x)
        else:
            return predict_tree(tree["right"], x)

def predict(trees, X):
    predictions = []
    for i in range(len(X)):
        p = np.mean([predict_tree(tree, X[i]) for tree in trees])
        predictions.append(p)
    return np.array(predictions)

def draw_bootstrap(X,y):
    n_samp = len(X)
    bootstrap_indices = np.random.choice(range(n_samp), n_samp)
    X_bootstrap = X[bootstrap_indices]
    y_bootstrap = y[bootstrap_indices]
    return X_bootstrap, y_bootstrap

def random_forest(X,y,n_estimators=5,max_depth=3):
    trees = []
    for i in range(n_estimators):
        X_bootstrap,y_bootstrap = draw_bootstrap(X,y)
        tree = build_tree(X_bootstrap, y_bootstrap, max_depth)
        trees.append(tree)
    return trees

def build_tree(X,y,max_depth):
    root = split_data(X,y)
    split_node(root, max_depth, depth=1)
    return root

def linear_regression(X,y):
    '''
    y = X . a + b
    '''
    length = len(y)
    if length == 0:
        return 0,0,float("inf")
    if len(X.shape) == 1:
        X = X.reshape(length,1)
    z = np.ones((length,1))
    m = np.concatenate((z,X),axis=1)
    ba = np.linalg.pinv(m) @ y
    b = ba[0]
    a = ba[1:]
    rss_tmp = m@ba - y
    rss = np.sum(rss_tmp * rss_tmp)
    return a,b,rss

def split_data(X,y):
    n_features = len(X[0])
    lowest_rss = float("inf")
    parent_a,parent_b,parent_rss = linear_regression(X,y)
    node = {"rss": parent_rss,
            "left": None,
            "right": None,
            "A": parent_a,
            "B": parent_b}
    child_len = -1
    n_splits = 100
    for feature_idx in range(n_features):
        arr = X[:,feature_idx]
        for split_point in np.linspace(np.min(arr), np.max(arr), n_splits):
            left = {"X_bootstrap":[], "y_bootstrap":[]}
            right = {"X_bootstrap":[], "y_bootstrap":[]}
            for i,value in enumerate(X[:,feature_idx]):
                if value <= split_point:
                    left["X_bootstrap"].append(X[i])
                    left["y_bootstrap"].append(y[i])
                else:
                    right["X_bootstrap"].append(X[i])
                    right["y_bootstrap"].append(y[i])
            left["X_bootstrap"] = np.array(left["X_bootstrap"])
            left["y_bootstrap"] = np.array(left["y_bootstrap"])
            right["X_bootstrap"] = np.array(right["X_bootstrap"])
            right["y_bootstrap"] = np.array(right["y_bootstrap"])
            _,_,rss_l = linear_regression(left["X_bootstrap"], left["y_bootstrap"])
            _,_,rss_r = linear_regression(right["X_bootstrap"], right["y_bootstrap"])
            if lowest_rss > rss_l + rss_r:
                lowest_rss = rss_l + rss_r
                node["rss"] = lowest_rss
                node["left"] = left
                node["right"] = right
                node["feature_idx"] = feature_idx
                node["split_point"] = split_point
                child_len = min(len(left["y_bootstrap"]), len(right["y_bootstrap"]))
    if lowest_rss / parent_rss > 0.90 or child_len <= 10:
        node["left"] = None
        node["right"] = None
    return node

def split_node(node, max_depth, depth):
    left = node["left"]
    right = node["right"]
    del(node["left"])
    del(node["right"])
    if left is None or depth > max_depth:
        node["left"] = None
        node["right"] = None
        return
    else:
        node["left"] = split_data(left["X_bootstrap"], left["y_bootstrap"])
        split_node(node["left"], max_depth, depth+1)
        node["right"] = split_data(right["X_bootstrap"], right["y_bootstrap"])
        split_node(node["right"], max_depth, depth+1)

def gen_data(n_samples, n_features, n_categories, is_random=False):
    n_range = 10
    if is_random:
        X = np.random.choice(range(n_range), n_samples * n_features).reshape(n_samples, n_features)
        y = np.random.choice(range(n_categories), n_samples)
        return X,y
    else:
        X,y2 = make_blobs(n_samples=n_samples, centers=n_categories, n_features=n_features, random_state=42)

        a = []
        b = []
        y = []
        for i in range(n_categories):
            a.append(np.random.choice(range(-5,6), n_features))
            b.append(10* np.random.normal())
        for i in range(n_samples):
            y.append(X[i] @ a[y2[i]] + b[y2[i]])
        y = np.array(y)
        f = np.vectorize(lambda x: round(x*2))
        X = f(X)
        return X,y
            
def main():
    n_samples = 1000
    n_features = 10
    n_range = 10
    n_categories = 4

    X,y = gen_data(n_samples, n_features, n_categories, is_random=False)
    model = random_forest(X, y, n_estimators=5, max_depth=3)
    preds = predict(model,X)

    y_mean = np.mean(y)
    tss = np.sum((y-y_mean)*(y-y_mean))
    rss = np.sum((y-preds) * (y-preds))
    print(f"R^2 = {1-rss/tss}")

    return

main()
