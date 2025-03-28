import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from scipy.signal import stft, get_window
from collections import Counter, defaultdict
import time
import warnings
warnings.filterwarnings("ignore")

# if GPU is available
# import cudf as cu
# from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
# from cuml.preprocessing import train_test_split as cutrain_test_split

DATA_PATH = "/home/xxx/Downloads/"

class MyRandomForestClassifier:
    def __init__(self, n_estimators=100, max_features="sqrt", max_depth=None,
                 min_samples_split=2, random_state=None, oob_score=False):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.oob_score = oob_score
        self.classes_ = None
        self.trees = []
        self.oob_indices = []
        self.oob_score_ = None
        self.rng = np.random.RandomState(random_state)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.trees = []
        self.oob_indices = []

        for _ in range(self.n_estimators):
            # Bootstrap sample
            indices = self.rng.choice(n_samples, size=n_samples, replace=True)
            oob = np.setdiff1d(np.arange(n_samples), indices)
            X_sample = X[indices]
            y_sample = y[indices]
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=self.rng.randint(0, 1e6)
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            self.oob_indices.append(oob)

        if self.oob_score:
            self._compute_oob_score(X, y)

    def _compute_oob_score(self, X, y):
        n_samples = X.shape[0]
        oob_votes = defaultdict(list)

        for tree, oobs in zip(self.trees, self.oob_indices):
            if len(oobs) == 0:
                continue
            preds = tree.predict(X[oobs])
            for oob, pred in zip(oobs, preds):
                oob_votes[oob].append(pred)

        correct = 0
        oob_count = 0
        for oob, votes in oob_votes.items():
            majority = Counter(votes).most_common(1)[0][0]
            if majority == y[oob]:
                correct += 1
            oob_count += 1

        self.oob_score_ = correct / oob_count if oob_count > 0 else None

    def predict2(self, X):
        # Leave old predict() code here.
        # This is based on majority vote and struggles when max_depth is small
        tree_preds = np.array([
            tree.predict(X) for tree in self.trees
        ])
        majority_votes = np.apply_along_axis(
            lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=tree_preds
        )
        return majority_votes

    def predict(self, X):
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        predictions = self.classes_[class_indices]
        return predictions
    
    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        cls2idx = {cls:idx for idx,cls in enumerate(self.classes_)}
        proba_sum = np.zeros([n_samples, n_classes])
        for tree in self.trees:
            tree_proba = tree.predict_proba(X)
            tree_classes = tree.classes_  # Each tree should store the classes it saw
            for idx,cls in enumerate(tree_classes):
                col = cls2idx[cls]
                proba_sum[:,col] += tree_proba[:,idx]
        return proba_sum / self.n_estimators

def test_MyRandomForestClassifier():
    N = 10000
    p = 100
    X = np.random.normal(size=[N,p])
    y = np.random.choice(14, size=N)
    accs1 = []
    oobs1 = []
    accs2 = []
    oobs2 = []
    ns = []
    max_depth = 5
    for n in range(10,201,10):
        model1 = MyRandomForestClassifier(n_estimators=n, max_depth=max_depth, oob_score=True, random_state=42)
        model2 = RandomForestClassifier(n_estimators=n, max_depth=max_depth, oob_score=True, random_state=42)
        model1.fit(X, y)
        model2.fit(X, y)
        y_pred1 = model1.predict(X)
        y_pred2 = model2.predict(X)
        acc1 = np.sum(y == y_pred1) / N
        acc2 = np.sum(y == y_pred2) / N
        accs1.append(acc1)
        accs2.append(acc2)
        oobs1.append(model1.oob_score_)
        oobs2.append(model2.oob_score_)
        ns.append(n)
    plt.plot(ns, accs1, "o-", label="Accuracy Mine")
    plt.plot(ns, oobs1, "o-", label="OOB Mine")
    plt.plot(ns, accs2, "o-", label="Accuracy SK")
    plt.plot(ns, oobs2, "o-", label="OOB SK")
    plt.legend()
    plt.xlabel("number of estimators")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. n_estimators")
    plt.show()

def c_factor(n):
    EULER_CONSTANT = 0.5772156649
    if n <= 1:
        return 0
    return 2 * (np.log(n - 1) + EULER_CONSTANT) - 2 * (n - 1) / n

class MyIsolationTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X):
        self.n_samples = X.shape[0]
        self._fit(X, 0)

    def _fit(self, X, current_depth):
        if current_depth >= self.max_depth or len(X) <= 1:
            self.size = len(X)
            return None

        # Random feature and split
        feature_idx = np.random.randint(0, X.shape[1])
        min_val, max_val = np.min(X[:, feature_idx]), np.max(X[:, feature_idx])
        if min_val == max_val:
            self.size = len(X)
            return None

        split_val = np.random.uniform(min_val, max_val)

        self.feature_idx = feature_idx
        self.split_val = split_val

        left_mask = X[:, feature_idx] < split_val
        right_mask = ~left_mask

        self.left = MyIsolationTree(self.max_depth)
        self.left._fit(X[left_mask], current_depth + 1)

        self.right = MyIsolationTree(self.max_depth)
        self.right._fit(X[right_mask], current_depth + 1)

    def path_length(self, x, current_depth=0):
        if hasattr(self, "size"):
            return current_depth + c_factor(self.size)

        if x[self.feature_idx] < self.split_val:
            return self.left.path_length(x, current_depth + 1)
        else:
            return self.right.path_length(x, current_depth + 1)

class MyIsolationForest:
    def __init__(self, n_trees=100, max_samples=256, contamination=0.1):
        self.n_trees = n_trees
        self.max_samples = max_samples
        self.contamination = contamination
        self.trees = []

    def fit(self, X):
        self.X = X
        self.trees = []
        self.sample_size = min(self.max_samples, len(X))
        self.c = c_factor(self.sample_size)

        for _ in range(self.n_trees):
            sample_indices = np.random.choice(len(X), self.sample_size, replace=False)
            X_sample = X[sample_indices]
            tree = MyIsolationTree(max_depth=int(np.ceil(np.log(self.sample_size) / np.log(2))))
            tree.fit(X_sample)
            self.trees.append(tree)

    def anomaly_score(self, X):
        scores = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            path_lengths = np.array([tree.path_length(x) for tree in self.trees])
            scores[i] = 2 ** (-np.mean(path_lengths) / self.c)
        return scores

    def predict(self, X):
        ano_scores = self.anomaly_score(X)
        threshold = np.percentile(ano_scores, 100 * (1 - self.contamination))
        predictions = np.where(ano_scores < threshold, 1, -1).astype(int)
        return predictions

def test_MyIsolationForest():
    seed = 1
    mean1 = np.array([2, 2])
    mean2 = np.array([-2, -2])
    cov = np.array([[1, 0], [0, 1]])
    norm1 = np.random.multivariate_normal(mean1, cov, size=100)
    norm2 = np.random.multivariate_normal(mean2, cov, size=100)
    lower, upper = -10, 10
    anom = (upper - lower)*np.random.rand(10, 2) + lower

    df = np.vstack([norm1, norm2, anom])
    df = pd.DataFrame(df, columns=["feat1", "feat2"])

    contamination = 0.1
    model1 = IsolationForest(n_estimators=100, random_state=seed, contamination=contamination)
    model1.fit(df)
    df["predict1"] = model1.predict(df)

    model2 = MyIsolationForest(n_trees=100, max_samples=256, contamination=contamination)
    model2.fit(df[["feat1","feat2"]].values)
    df["predict2"] = model2.predict(df[["feat1", "feat2"]].values)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    sns.scatterplot(x="feat1", y="feat2", data=df, hue="predict1", palette="bright", ax=ax[0])
    ax[0].set_title("Anomaly Prediction: sklearn IsolationForest")
    sns.scatterplot(x="feat1", y="feat2", data=df, hue="predict2", palette="bright", ax=ax[1])
    ax[1].set_title("Anomaly Prediction: Custom IsolationForest")
    plt.tight_layout()
    plt.show()
      
def accuracy_comp(X_train, y_train, X_test, y_test, model_name, max_depth, min_samples_split, seed, is_gpu=False):
    accs = []
    oobs = []
    ks = []
    for k in [2,3,4,5,6,7,8] + list(range(10,101,10)):
        if is_gpu == True:
            model = model_name(n_estimators=k, max_depth=max_depth, min_samples_split=min_samples_split, n_bins=10, random_state=seed)
        else:
            model = model_name(n_estimators=k, max_depth=max_depth, min_samples_split=min_samples_split, oob_score=True, random_state=seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = np.sum(y_test == y_pred) / len(y_test)
        accs.append(acc)
        if is_gpu == False:
            oobs.append(model.oob_score_)
        ks.append(k)
    return accs,oobs,ks

def draw_graph(X,y,model,seed,is_gpu=False):
    accs05 = dict()
    accs10 = dict()
    accs15 = dict()
    accs20 = dict()
    oobs05 = dict()
    oobs10 = dict()
    oobs15 = dict()
    oobs20 = dict()
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5,random_state=seed, stratify=y)
    for mss in [2,4,8,16]:
        a05,o05,ks = accuracy_comp(X_train, y_train, X_test, y_test, model, max_depth=5,  min_samples_split=mss, seed=seed, is_gpu=is_gpu)
        a10,o10,_  = accuracy_comp(X_train, y_train, X_test, y_test, model, max_depth=10, min_samples_split=mss, seed=seed, is_gpu=is_gpu)
        a15,o15,_  = accuracy_comp(X_train, y_train, X_test, y_test, model, max_depth=15, min_samples_split=mss, seed=seed, is_gpu=is_gpu)
        a20,o20,_  = accuracy_comp(X_train, y_train, X_test, y_test, model, max_depth=200, min_samples_split=mss, seed=seed, is_gpu=is_gpu)
        accs05[mss] = a05
        accs10[mss] = a10
        accs15[mss] = a15
        accs20[mss] = a20
        oobs05[mss] = o05
        oobs10[mss] = o10
        oobs15[mss] = o15
        oobs20[mss] = o20
    xmin = 0
    xmax = 105
    fig,ax = plt.subplots(2,2,figsize=(8,8),sharex=True,sharey=True)
    mss = 2
    ax[0,0].plot(ks, accs05[mss], "o-", label="max_depth=5")
    ax[0,0].plot(ks, accs10[mss], "o-", label="max_depth=10")
    ax[0,0].plot(ks, accs15[mss], "o-", label="max_depth=15")
    ax[0,0].plot(ks, accs20[mss], "o-", label="max_depth=200")
    ax[0,0].legend()
    ax[0,0].set_xlim([xmin,xmax])
    ax[0,0].set_ylim([0,1])
    ax[0,0].set_title(f"max_samples_split={mss}")
    mss = 4
    ax[0,1].plot(ks, accs05[mss], "o-", label="max_depth=5")
    ax[0,1].plot(ks, accs10[mss], "o-", label="max_depth=10")
    ax[0,1].plot(ks, accs15[mss], "o-", label="max_depth=15")
    ax[0,1].plot(ks, accs20[mss], "o-", label="max_depth=200")
    ax[0,1].legend()
    ax[0,1].set_xlim([xmin,xmax])
    ax[0,1].set_ylim([0,1])
    ax[0,1].set_title(f"max_samples_split={mss}")
    mss = 8
    ax[1,0].plot(ks, accs05[mss], "o-", label="max_depth=5")
    ax[1,0].plot(ks, accs10[mss], "o-", label="max_depth=10")
    ax[1,0].plot(ks, accs15[mss], "o-", label="max_depth=15")
    ax[1,0].plot(ks, accs20[mss], "o-", label="max_depth=200")
    ax[1,0].legend()
    ax[1,0].set_xlim([xmin,xmax])
    ax[1,0].set_ylim([0,1])
    ax[1,0].set_title(f"max_samples_split={mss}")
    mss = 16
    ax[1,1].plot(ks, accs05[mss], "o-", label="max_depth=5")
    ax[1,1].plot(ks, accs10[mss], "o-", label="max_depth=10")
    ax[1,1].plot(ks, accs15[mss], "o-", label="max_depth=15")
    ax[1,1].plot(ks, accs20[mss], "o-", label="max_depth=200")
    ax[1,1].legend()
    ax[1,1].set_xlim([xmin,xmax])
    ax[1,1].set_ylim([0,1])
    ax[1,1].set_title(f"max_samples_split={mss}")
    plt.show()

    if is_gpu == True:
        return

    fig,ax = plt.subplots(2,2,figsize=(8,8),sharex=True,sharey=True)
    mss = 2
    ax[0,0].plot(ks, oobs05[mss], "o-", label="max_depth=5")
    ax[0,0].plot(ks, oobs10[mss], "o-", label="max_depth=10")
    ax[0,0].plot(ks, oobs15[mss], "o-", label="max_depth=15")
    ax[0,0].plot(ks, oobs20[mss], "o-", label="max_depth=200")
    ax[0,0].legend()
    ax[0,0].set_xlim([xmin,xmax])
    ax[0,0].set_ylim([0,1])
    ax[0,0].set_title(f"max_samples_split={mss}")
    mss = 4
    ax[0,1].plot(ks, oobs05[mss], "o-", label="max_depth=5")
    ax[0,1].plot(ks, oobs10[mss], "o-", label="max_depth=10")
    ax[0,1].plot(ks, oobs15[mss], "o-", label="max_depth=15")
    ax[0,1].plot(ks, oobs20[mss], "o-", label="max_depth=200")
    ax[0,1].legend()
    ax[0,1].set_xlim([xmin,xmax])
    ax[0,1].set_ylim([0,1])
    ax[0,1].set_title(f"max_samples_split={mss}")
    mss = 8
    ax[1,0].plot(ks, oobs05[mss], "o-", label="max_depth=5")
    ax[1,0].plot(ks, oobs10[mss], "o-", label="max_depth=10")
    ax[1,0].plot(ks, oobs15[mss], "o-", label="max_depth=15")
    ax[1,0].plot(ks, oobs20[mss], "o-", label="max_depth=200")
    ax[1,0].legend()
    ax[1,0].set_xlim([xmin,xmax])
    ax[1,0].set_ylim([0,1])
    ax[1,0].set_title(f"max_samples_split={mss}")
    mss = 16
    ax[1,1].plot(ks, oobs05[mss], "o-", label="max_depth=5")
    ax[1,1].plot(ks, oobs10[mss], "o-", label="max_depth=10")
    ax[1,1].plot(ks, oobs15[mss], "o-", label="max_depth=15")
    ax[1,1].plot(ks, oobs20[mss], "o-", label="max_depth=200")
    ax[1,1].legend()
    ax[1,1].set_xlim([xmin,xmax])
    ax[1,1].set_ylim([0,1])
    ax[1,1].set_title(f"max_samples_split={mss}")
    plt.show()

def draw_ROC(y, y_pred, classes, name=""):
    n_classes = len(classes)
    y_bin = label_binarize(y, classes=np.arange(n_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr["micro"], tpr["micro"], label=f"micro-average ROC (area = {roc_auc["micro"]:.2f})", linewidth=2)
    plt.plot(fpr["macro"], tpr["macro"], label=f"macro-average ROC (area = {roc_auc["macro"]:.2f})", linewidth=2)
    if True:
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=1, alpha=0.3, label=f"Class {classes[i]} (area = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Multi-class ROC Curve for {name}")
    plt.legend(loc="lower right", fontsize="small")
    plt.grid(True)
    plt.show()

def stellar_classification():
    seed = 10
    filename = f"{DATA_PATH}/stellar_classification_dataset/star_classification.csv"
    df = pd.read_csv(filename)
    y_col = "class"
    X_col = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]

    df_X = df[X_col]
    y = df[y_col].values

    df_X = pd.get_dummies(df_X, columns=[c for c in df_X.columns if df_X[c].dtype == "object"], drop_first=True)
    X = df_X.values
    le = LabelEncoder()
    y = le.fit_transform(df[y_col].values)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(f"data size = {X.shape}")
    
    model = cuRandomForestClassifier(n_estimators=200, max_depth=100, min_samples_split=2, n_bins=10, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5,random_state=seed, stratify=y)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)
    draw_ROC(y_test, y_pred, le.classes_, name="Stellar")

    draw_graph(X,y,cuRandomForestClassifier,seed,True)

def spanish_wine():
    seed = 1
    filename = f"{DATA_PATH}/spanish_wine_quality_dataset/wines_SPA.csv"
    df = pd.read_csv(filename)

    # fix missing values
    df["type"] = df["type"].fillna("NoType")
    df["body"] = df["body"].fillna(0).astype(int)
    df["acidity"] = df["acidity"].fillna(0).astype(int)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    mean_year = round(df["year"].mean())
    df["year"] = df["year"].fillna(mean_year).astype(int)
    
    y_col = "type"
    X_col = [col for col in df.columns if col != y_col]
    df_X = df[X_col]
    df_X = pd.get_dummies(df_X, columns=[c for c in df_X.columns if df_X[c].dtype == "object"], drop_first=True)
    X = df_X.values
    le = LabelEncoder()
    y = le.fit_transform(df[y_col].values)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if False:
        threshold = 0.1
        correlations = np.array([pearsonr(X[:,i], y)[0] for i in range(X.shape[1])])
        print(correlations)
        mask = np.abs(correlations) > threshold
        X_filtered = X[:, mask]
        for i, corr in enumerate(correlations):
            if mask[i]:
                print(f"Column {i:2d} | Corr = {corr:.3f} | {"Kept" if mask[i] else "Dropped"}")
        return
    
    print(f"data size = {X.shape}")
    
    model = MyRandomForestClassifier(n_estimators=200, max_depth=100, min_samples_split=2, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5,random_state=seed, stratify=y)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)
    draw_ROC(y_test, y_pred, le.classes_, name="Spanish Wine")

    draw_graph(X,y,MyRandomForestClassifier,seed,False)

def mental_health():
    seed = 1
    filename = f"{DATA_PATH}/mental_health_and_lifestyle_habits/Mental_Health_Lifestyle_Dataset.csv"   
    df = pd.read_csv(filename)
    df["Mental Health Condition"] = df["Mental Health Condition"].fillna("No issue")
    y_col = "Mental Health Condition"
    X_col = [col for col in df.columns if col != y_col]
    df_X = df[X_col]
    df_X = pd.get_dummies(df_X, columns=[c for c in df_X.columns if df_X[c].dtype == "object"], drop_first=True)
    X = df_X.values
    le = LabelEncoder()
    y = le.fit_transform(df[y_col].values)

    if False:
        y2 = y * 2.1 + 3 + np.random.random(size=len(y))*1
        X = np.hstack([X,y2.reshape(-1,1)])
        threshold = 0.1
        correlations = np.array([pearsonr(X[:,i], y)[0] for i in range(X.shape[1])])
        mask = np.abs(correlations) > threshold
        X_filtered = X[:, mask]
        for i, corr in enumerate(correlations):
            if mask[i]:
                print(f"Column {i:2d} | Corr = {corr:.3f} | {"Kept" if mask[i] else "Dropped"}")
        pca = PCA()
        pca.fit(X)
        s = pca.explained_variance_ratio_
        plt.plot(np.arange(len(s)), np.cumsum(s), "o-")
        plt.grid()
        plt.show()
        return

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(f"data size = {X.shape}")

    model = MyRandomForestClassifier(n_estimators=200, max_depth=100, min_samples_split=2, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5,random_state=seed, stratify=y)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)
    draw_ROC(y_test, y_pred, le.classes_, name="Mental Health")

    draw_graph(X,y,MyRandomForestClassifier,seed,False)

def indiana_crime():
    seed = 1
    filename1 = "/home/ei/Downloads/indiana_crime_analysis/2023_q1.csv"
    filename2 = "/home/ei/Downloads/indiana_crime_analysis/2023_q2.csv"
    filename3 = "/home/ei/Downloads/indiana_crime_analysis/2023_q3.csv"
    filename4 = "/home/ei/Downloads/indiana_crime_analysis/2023_q4.csv"
    df1 = pd.read_csv(filename1)
    df2 = pd.read_csv(filename2)
    df3 = pd.read_csv(filename3)
    df4 = pd.read_csv(filename4)
    df = pd.concat([df1, df2, df3, df4], ignore_index=True)

    df["COUNTY"] = df["COUNTY"].fillna("NoData")
    df["month_sin"] = np.sin(2 * np.pi * df["MONTH"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["MONTH"] / 12)
    #df = df.drop(columns=["MONTH"])
    y_col = "CHARGE"
    X_col = ["AGE_GROUP", "RACE", "SEX", "COUNTY", "month_sin", "month_cos"]
    df_X = df[X_col]
    df_X = pd.get_dummies(df_X, columns=[c for c in df_X.columns if df_X[c].dtype == "object"], drop_first=True)
    X = df_X.values
    le = LabelEncoder()
    y = le.fit_transform(df[y_col].values)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(f"data size = {X.shape}")

    if False:
        threshold = 0.1
        column_name = list(df_X.columns)
        correlations = np.array([pearsonr(X[:,i], y)[0] for i in range(X.shape[1])])
        mask = np.abs(correlations) > threshold
        X_filtered = X[:, mask]
        for i, corr in enumerate(correlations):
            print(f"Column {column_name[i]} | Corr = {corr:.3f} | {"Kept" if mask[i] else "Dropped"}")
        return

    model = cuRandomForestClassifier(n_estimators=200, max_depth=100, min_samples_split=2, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5,random_state=seed, stratify=y)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)
    draw_ROC(y_test, y_pred, le.classes_, name="Indiana Crime")
    
    draw_graph(X,y,cuRandomForestClassifier,seed,True)

def heart_attack_prediction():
    seed = 1
    filename = f"{DATA_PATH}/heart_attack_prediction/heart_attack_dataset.csv"
    df = pd.read_csv(filename)
    y_col = "Outcome"
    X_col = [col for col in df.columns if col != y_col]
    df_X = df[X_col]
    df_X = pd.get_dummies(df_X, columns=[c for c in df_X.columns if df_X[c].dtype == "object"], drop_first=True)
    X = df_X.values
    le = LabelEncoder()
    y = le.fit_transform(df[y_col].values)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(f"data size = {X.shape}")

    if False:
        threshold = 0.1
        column_name = list(df_X.columns)
        correlations = np.array([pearsonr(X[:,i], y)[0] for i in range(X.shape[1])])
        mask = np.abs(correlations) > threshold
        X_filtered = X[:, mask]
        for i, corr in enumerate(correlations):
            print(f"Column {i:2d} | Corr = {corr:.3f} | {"Kept" if mask[i] else "Dropped"}")
        pca = PCA()
        pca.fit(X)
        s = pca.explained_variance_ratio_
        plt.plot(np.arange(len(s)), np.cumsum(s), "o-")
        plt.grid()
        plt.show()
        return
    draw_graph(X,y,cuRandomForestClassifier,seed,True)

def huntington_disease():
    seed = 1
    filename = f"{DATA_PATH}/huntington_disease_dataset/Huntington_Disease_Dataset.csv"
    df = pd.read_csv(filename)
    
    if False:
        y_col = "Category"
        X_col = ["Age", "Sex", "Family_History", "HTT_CAG_Repeat_Length", "Motor_Symptoms", "Cognitive_Decline", "Chorea_Score",
                 "Brain_Volume_Loss", "Functional_Capacity", "Gene_Mutation_Type", "HTT_Gene_Expression_Level", "Protein_Aggregation_Level",
                 "Disease_Stage", "Gene/Factor", "Chromosome_Location", "Function", "Effect"]
        df_X = df[X_col]
        df_X = pd.get_dummies(df_X, columns=[c for c in df_X.columns if df_X[c].dtype == "object"], drop_first=True)
        X = df_X.values
        le = LabelEncoder()
        y = le.fit_transform(df[y_col].values)
        columns = list(df_X.columns)
        threshold = 0.1
        correlations = np.array([pearsonr(X[:,i], y)[0] for i in range(X.shape[1])])
        mask = np.abs(correlations) > threshold
        X_filtered = X[:, mask]
        for i, corr in enumerate(correlations):
            if mask[i]:
                print(f"Column {i:2d} {columns[i]} | Corr = {corr:.3f} | {"Kept" if mask[i] else "Dropped"}")
        pca = PCA()
        pca.fit(X)
        s = pca.explained_variance_ratio_
        plt.plot(np.arange(len(s)), np.cumsum(s), "o-")
        plt.grid()
        plt.show()
        return

    y_col = "Category"
    X_col = ["Gene/Factor", "Chromosome_Location", "Function", "Effect"]
    df_X = df[X_col]
    df_X = pd.get_dummies(df_X, columns=[c for c in df_X.columns if df_X[c].dtype == "object"], drop_first=True)
    X = df_X.values
    le = LabelEncoder()
    y = le.fit_transform(df[y_col].values)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(f"data size = {X.shape}")
    draw_graph(X,y,cuRandomForestClassifier,seed,True)

def credit_fraud_detection():
    seed = 1
    filename = f"{DATA_PATH}/credit_card_fraud_detection/creditcard.csv"
    df = pd.read_csv(filename)    
    X_col = ["Time", "V1",  "V2",  "V3",  "V4",  "V5",  "V6",  "V7",  "V8",  "V9", "V10",
             "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
             "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"]
    df_X = df[X_col]

    # Original: Counter({0: 284315, 1: 492})
    # To match IsoForest: 0 => 1, 1 => -1
    y_true = df["Class"].replace({0: 1, 1: -1})
    accs = []
    f1s1 = []
    f1s2 = []
    ks = []
    for k in [5,10,15,20,25,30,50,80] + list(range(100,1001,100)):
        model1 = IsolationForest(n_estimators=k, random_state=seed, bootstrap=True, contamination=0.00172)
        model1.fit(df_X)
        y_pred1 = model1.predict(df_X)
        
        model2 = IsolationForest(n_estimators=k, random_state=seed, bootstrap=False, contamination=0.00172)
        model2.fit(df_X)
        y_pred2 = model2.predict(df_X)
        
        ks.append(k)
        f1s1.append(f1_score(y_true, y_pred1))
        f1s2.append(f1_score(y_true, y_pred2))
    
    plt.plot(ks, f1s1, label="Bootstrap ON")
    plt.plot(ks, f1s2, label="Bootstrap OFF")
    plt.legend()
    plt.xlabel("Number of Trees")
    plt.ylabel("F1-Score")
    plt.title("F1-Score vs. Number of Trees")
    plt.show()

    model3 = IsolationForest(n_estimators=200, random_state=seed, bootstrap=True, contamination=0.00172)
    model3.fit(df_X)
    y_pred = model3.predict(df_X)

    cm = confusion_matrix(y_true, y_pred)
    labels = ["True Negative (TN)", "False Positive (FP)", 
              "False Negative (FN)", "True Positive (TP)"]
    cm_reshaped = cm.reshape(-1)
    annot = np.array([f"{label}\n{value}" for label, value in zip(labels, cm_reshaped)]).reshape(2, 2)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False,
                xticklabels=["Predicted -1", "Predicted 1"],
                yticklabels=["Actual -1", "Actual 1"])
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix with Labels")
    plt.tight_layout()
    plt.show()
    
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))

    model4 = IsolationForest(n_estimators=200, random_state=seed, bootstrap=True, contamination=0.00172)
    X_train, X_test, y_train, y_test = train_test_split(df_X,y_true,test_size=0.5,random_state=seed, stratify=y_true)
    model4.fit(X_train)

    scores = model4.decision_function(X_test)  # higher = more normal
    y_pred = model4.predict(X_test)            # -1 = outlier, 1 = inlier

    # Convert the ground truth labels: 1 for anomaly, 0 for normal
    y_true = (y_test == -1).astype(int)

    fpr, tpr, thresholds = roc_curve(y_true, -scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"Isolation Forest ROC (AUC = {roc_auc:.2f})", color="darkred")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve of Credit Fraud Detection Dataset")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def mystft(signal, window_size, sr=1):
    window = get_window("hann", window_size)
    step = 1
    n_segments = (len(signal) - window_size) // step + 1
    segments = np.lib.stride_tricks.sliding_window_view(signal, window_shape=window_size)[::step]
    segments_zero_mean = segments - segments.mean(axis=1, keepdims=True)
    windowed = segments_zero_mean * window
    Zxx = np.fft.rfft(windowed, axis=1)
    power = np.abs(Zxx) ** 2
    f = np.fft.rfftfreq(window_size, d=1/sr)
    t = np.arange(n_segments) * step / sr
    return power.T, t, f

def bitcoin_histrical_data():
    seed = 1
    filename = f"{DATA_PATH}/bitcoin_histrical_data/btcusd_1-min_data.csv"
    df = pd.read_csv(filename)
    df["Datetime"] = pd.to_datetime(df["Timestamp"], unit="s")

    x = df["Close"].values
    time_index = df["Datetime"].values

    sr = 1/60
    powers, times, freqs = mystft(x, 60, sr)
    times /= 3600*24  # Convert to days
    freqs *= 3600     # Convert to cycles/hour

    contami_rate = 100 / powers.shape[1]
    model = IsolationForest(n_estimators=200, random_state=seed, contamination=contami_rate)
    outliers = model.fit_predict(powers.T)

    window_size = 60
    step = 1
    num_windows = powers.shape[1]
    center_indices = np.arange(num_windows) * step + window_size // 2
    center_indices = center_indices[center_indices < len(df)]
    window_centers = df["Datetime"].iloc[center_indices].values

    start_date = pd.to_datetime("2025-01-20")
    end_date = pd.to_datetime("2025-01-21")
    df_period = df[(df["Datetime"] >= start_date) & (df["Datetime"] <= end_date)].copy()
    mask_period = (window_centers >= start_date) & (window_centers <= end_date)
    outlier_indices_period = np.where((outliers == -1) & mask_period)[0]

    plt.figure(figsize=(14, 6))
    plt.plot(df_period["Datetime"], df_period["Close"], label="Bitcoin Price (Selected Period)")
    for idx in outlier_indices_period:
        if idx < len(window_centers):
            plt.axvline(x=window_centers[idx], color="red", linestyle="--", alpha=0.2)
    plt.title("Bitcoin Price with Anomaly Markers (2025-01-15 to 2025-02-15)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

def bitcoin_histrical_data_entire_period():
    seed = 1
    filename = f"{DATA_PATH}/bitcoin_histrical_data/btcusd_1-min_data.csv"
    df = pd.read_csv(filename)
    df["Datetime"] = pd.to_datetime(df["Timestamp"], unit="s")
    x = df["Close"].values
    time_index = df["Datetime"].values

    sr = 1/60
    powers, times, freqs = mystft(x, 60, sr)
    times /= 3600*24  # Convert to days
    freqs *= 3600     # Convert to cycles/hour

    # powers: (31, 6944741)
    contami_rate = 100 / powers.shape[1]
    model = IsolationForest(n_estimators=200, random_state=seed, contamination=contami_rate)
    outliers = model.fit_predict(powers.T)

    window_size = 60
    step = 1
    num_windows = powers.shape[1]
    center_indices = np.arange(num_windows) * step + window_size // 2
    center_indices = center_indices[center_indices < len(df)]  # Make sure it"s safe
    window_centers = df["Datetime"].iloc[center_indices].values
    plt.figure(figsize=(14, 6))
    plt.plot(time_index, x, label="Bitcoin Price")
    outlier_indices = np.where(outliers == -1)[0]
    print(outlier_indices)
    for idx in outlier_indices:
        if idx < len(window_centers):  # Just in case
            print(f"idx={idx} window_centers[{idx}]={window_centers[idx]}")
            plt.axvline(x=window_centers[idx], color="red", linestyle="--", alpha=0.5)
    plt.title("Bitcoin Price with Anomaly Markers")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def indiana_crime2():
    sns.set_theme(style="whitegrid")
    data = {
        "Traffic": 200824,
        "Drug": 169945,
        "Violent": 121615,
        "Procedural": 105529,
        "Property": 93174, 
        "Other": 38040,
        "Fraud": 17907, 
        "Child": 13478,
        "Alcohol": 12147,
        "Firearm": 11512,
        "Sex": 10772
    }
    sorted_items = sorted(data.items(), key=lambda item: item[1], reverse=False)
    labels, sizes = zip(*sorted_items)
    plt.figure(figsize=(8,8))
    plt.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(edgecolor="w")
    )
    plt.title("Distribution of Crime Categories")
    plt.tight_layout()
    plt.show()

def main():
    # test_MyRandomForestClassifier()
    # test_MyIsolationForest()
    mental_health()
    spanish_wine()
    stellar_classification()
    indiana_crime()
    indiana_crime2()
    huntington_disease()
    heart_attack_prediction()
    credit_fraud_detection()
    bitcoin_histrical_data_entire_period()
    bitcoin_histrical_data()

main()
