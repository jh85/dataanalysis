import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize
from sklearn.cluster import KMeans
from sklearn.svm import SVC, LinearSVC, SVR, LinearSVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.utils import class_weight
from sklearn.base import BaseEstimator, ClassifierMixin
import umap
import time
from collections import Counter

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

DATA_PATH = "/home/xxx/Downloads/"

def load_data_superconductor(chem_data=True, formula_data=True, scaler=None, y_data="label", seed=1):
    filename1  = f"{DATA_PATH}/critical-temperature-of-superconductors/train.csv"
    filename2 = f"{DATA_PATH}/critical-temperature-of-superconductors/formula_train.csv"

    df1_X, df1_y = None, None
    df2_X, df2_y = None, None

    if chem_data == True:
        df1 = pd.read_csv(filename1)
        columns1 = list(df1.columns)
        columns1.remove("critical_temp")
        df1_X = df1[columns1]
        df1_y = df1["critical_temp"]
    
    if formula_data == True:
        df2 = pd.read_csv(filename2)
        columns2 = list(df2.columns)
        columns2.remove("critical_temp")
        columns2.remove("material")
        df2_X = df2[columns2]
        df2_y = df2["critical_temp"]

    if chem_data and formula_data:
        assert df1_y.equals(df2_y), "Mismatch between target variables in df1 and df2"
        df_X = pd.concat([df1_X.reset_index(drop=True), df2_X.reset_index(drop=True)], axis=1)
        df_y = df1_y
    elif chem_data:
        df_X = df1_X
        df_y = df1_y
    elif formula_data:
        df_X = df2_X
        df_y = df2_y
    else:
        raise ValueError("wrong option")

    if scaler:
        X = scaler.transform(df_X)
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(df_X)
    y = df_y.values
    
    if y_data == "value":
        return X, y, scaler

    bin_edges = np.arange(0, 200, 10)
    bin_labels = [i for i in range(len(bin_edges)-1)]
    binned = pd.cut(y, bins=bin_edges, right=False, labels=bin_labels)
    bin_counts = binned.value_counts().sort_index()
    y_labels = pd.cut(y, bins=bin_edges, right=False, labels=bin_labels).to_numpy()

    # Duplicate entries in bins 14 and 18
    df_X = pd.DataFrame(X)
    df_y_labels = pd.Series(y_labels)
    # Bin 18 has 1 entry → duplicate 5 times (create 6 copies total)
    idx_bin18 = df_y_labels[df_y_labels == 18].index
    df_X_bin18 = pd.concat([df_X.loc[idx_bin18]] * 6, ignore_index=True)
    df_y_bin18 = pd.concat([df_y_labels.loc[idx_bin18]] * 6, ignore_index=True)

    # Bin 14 has 2 entries → duplicate each 3 times (total 6 copies)
    idx_bin14 = df_y_labels[df_y_labels == 14].index
    df_X_bin14 = pd.concat([df_X.loc[idx_bin14]] * 3, ignore_index=True)
    df_y_bin14 = pd.concat([df_y_labels.loc[idx_bin14]] * 3, ignore_index=True)

    df_X_augmented = pd.concat([df_X, df_X_bin18, df_X_bin14], ignore_index=True)
    df_y_augmented = pd.concat([df_y_labels, df_y_bin18, df_y_bin14], ignore_index=True)
    X_aug = df_X_augmented.values
    y_aug = df_y_augmented.values.reshape(-1)

    return X_aug, y_aug, scaler

    # 0~10       6158
    # 10~20      2277
    # 20~30      1486
    # 30~40      1299
    # 40~50       687
    # 50~60       637
    # 60~70       640
    # 70~80       981
    # 80~90      1256
    # 90~100      938
    # 100~110     252
    # 110~120     199
    # 120~130     113
    # 130~140      84
    # 140~150       2   => duplicate this for CV
    # 150~160       0   (no entry)
    # 160~170       0   (no entry)
    # 170~180       0   (no entry)
    # 180~190       1   => duplicate this for CV

def superconductor_critical_temp_dtc():
    seed = 1
    X,y,scaler = load_data_superconductor(chem_data=True,formula_data=False,scaler=None,y_data="label",seed=1)

    dtree = DecisionTreeClassifier()
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "random_state": [seed],
    }
    grid_search = GridSearchCV(estimator=dtree,
                               param_grid=param_grid,
                               cv=5,
                               n_jobs=-1,
                               scoring="accuracy",
                               verbose=0)
    # grid_search.fit(X, y)
    # print("Best Hyperparameters:", grid_search.best_params_)
    # Best Hyperparameters: {"criterion": "gini", "max_depth": None, "min_samples_leaf": 2, "min_samples_split": 2, "random_state": 1}
    best_params = {"criterion": "gini", "max_depth": None, "min_samples_leaf": 2, "min_samples_split": 2, "random_state": 1}
    dtree_best = DecisionTreeClassifier(**best_params)
    dtree_best.fit(X,y)
    y_pred = dtree_best.predict(X)
    print("Classification Report:")
    print(classification_report(y, y_pred))
    print("Accuracy Score:", accuracy_score(y, y_pred))

    y_pred_prob = dtree_best.predict_proba(X)
    draw_ROC(y, y_pred_prob, "DecisionTree")

def show_decision_boundaries(svc_model, X_hd, y):
    seed = 1
    umap_model = umap.UMAP(n_components=2, n_jobs=-1, random_state=seed)
    X_2d = umap_model.fit_transform(X_hd)

    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    n_grid = 50
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_grid),
                         np.linspace(y_min, y_max, n_grid))
    xxyy_2d = np.c_[xx.ravel(), yy.ravel()]
    xxyy_hd = umap_model.inverse_transform(xxyy_2d)
    Z = svc_model.predict(xxyy_hd)
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(2, 1, figsize=(8, 10), sharex=True, sharey=True)

    # Plot 1: Decision regions + boundary lines
    contourf = ax[0].contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    boundary_lines = ax[0].contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=1.5)
    ax[0].set_title("Decision Boundaries (UMAP + SVM)")
    ax[0].set_ylabel("Feature 2")

    # Plot 2: Data points only
    scatter = ax[1].scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors="k", s=10, alpha=0.8)
    ax[1].set_title("Data Points in 2D (UMAP Projection)")
    ax[1].set_xlabel("Feature 1")
    ax[1].set_ylabel("Feature 2")

    plt.tight_layout()
    plt.show()
    
def superconductor_critical_temp_SVC(model_type):
    seed = 1
    X,y,scaler = load_data_superconductor(chem_data=True,formula_data=False,scaler=None,y_data="label",seed=1)

    if model_type == "LinearSVC":
        param_grid = {
            "C": [1, 10, 100, 1000],
            "random_state": [seed],
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        svc = LinearSVC(max_iter=10000, random_state=seed)
        grid_search = GridSearchCV(estimator=svc,
                                   param_grid=param_grid,
                                   cv=cv,
                                   scoring="accuracy",
                                   n_jobs=-1,
                                   verbose=0)
        grid_search.fit(X, y)
        print("Best Hyperparameters:", grid_search.best_params_)
        print("Best Cross-Validated Accuracy:", grid_search.best_score_)
        # Best Hyperparameters: {"C": 100}
        # Best Cross-Validated Accuracy: 0.5424478586995345

    if model_type == "SVC":
        param_grid = {
            "C": [10, 100, 1000,10000,100000],
            "gamma": ["scale"],
            # "kernel": ["rbf","linear","poly"],
            "kernel": ["rbf"],
            "random_state": [seed],
        }
        cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=seed)
        svc = SVC()
        grid_search = GridSearchCV(estimator=svc,
                                   param_grid=param_grid,
                                   cv=cv,
                                   scoring="accuracy",
                                   n_jobs=-1,
                                   verbose=0)
        # takes 5 minutes
        # grid_search.fit(X, y)
        # print("Best Hyperparameters:", grid_search.best_params_)
        # print("Best Cross-Validated Accuracy:", grid_search.best_score_)
        # Best Hyperparameters: {"C": 10000, "gamma": "scale", "kernel": "rbf", "random_state": 1}
        # Best Cross-Validated Accuracy: 0.6444601104453062
        # Accuracy on full data: 0.7951474562331101
        # best_params = grid_search.best_params_
        best_params = {"C": 10000, "gamma": "scale", "kernel": "rbf", "random_state": 1}
        svc_best = SVC(**best_params)
        svc_best.fit(X, y)
        y_pred = svc_best.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"Accuracy on full data: {accuracy}")
        print("Classification Report:")
        print(classification_report(y, y_pred))

        svc_best_prob = SVC(**best_params, probability=True)
        svc_best_prob.fit(X,y)

    if model_type == "CSSVC":
        param_grid = {
            "C": [10, 100, 1000,10000,100000,1000000],
            "gamma": ["scale"],
            "kernel": ["rbf"],
            "random_state": [seed],
        }
        cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=seed)
        svc = CostSensitiveSVC()
        grid_search = GridSearchCV(estimator=svc,
                                   param_grid=param_grid,
                                   cv=cv,
                                   scoring="accuracy",
                                   n_jobs=-1,
                                   verbose=0)
        # grid_search.fit(X, y)
        # print("Best Hyperparameters:", grid_search.best_params_)
        # print("Best Cross-Validated Accuracy:", grid_search.best_score_)
        # best_params = grid_search.best_params_
        best_params = {"C": 100000, "gamma": "scale", "kernel": "rbf", "random_state": 1}
        # Best Hyperparameters: {"C": 100000, "gamma": "scale", "kernel": "rbf", "random_state": 1}
        # Best Cross-Validated Accuracy: 0.6338855598637058
        # Accuracy on full data: 0.8194101750675596
        svc_best = CostSensitiveSVC(**best_params)
        svc_best.fit(X, y)
        y_pred = svc_best.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"Accuracy on full data: {accuracy}")
        print("Classification Report:")
        print(classification_report(y, y_pred))

        svc_best_prob = CostSensitiveSVC(**best_params, probability=True)
        svc_best_prob.fit(X,y)

    if model_type == "FSVC":
        param_grid = {
            "C": [10, 100, 1000,10000,100000,1000000],
            "gamma": ["scale"],
            "kernel": ["rbf"],
            "epsilon": [1e-3, 1e-2, 1e-1],
            "random_state": [seed],
        }
        cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=seed)
        svc = FuzzySVC()
        grid_search = GridSearchCV(estimator=svc,
                                   param_grid=param_grid,
                                   cv=cv,
                                   scoring="accuracy",
                                   n_jobs=-1,
                                   verbose=0)
        # grid_search.fit(X, y)
        # print("Best Hyperparameters:", grid_search.best_params_)
        # print("Best Cross-Validated Accuracy:", grid_search.best_score_)
        # best_params = grid_search.best_params_
        best_params = {"C": 100000, "epsilon": 0.001, "gamma": "scale", "kernel": "rbf", "random_state": 1}
        # Best Hyperparameters: {"C": 100000, "epsilon": 0.001, "gamma": "scale", "kernel": "rbf", "random_state": 1}
        # Best Cross-Validated Accuracy: 0.6448713429679238
        # Accuracy on full data: 0.8301022206556221
        svc_best = FuzzySVC(**best_params)
        svc_best.fit(X, y)
        y_pred = svc_best.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"Accuracy on full data: {accuracy}")
        print("Classification Report:")
        print(classification_report(y, y_pred))

        svc_best_prob = FuzzySVC(**best_params, probability=True)
        svc_best_prob.fit(X,y)

    show_decision_boundaries(svc_best, X, y_pred)

    y_pred_prob = svc_best_prob.predict_proba(X)
    draw_ROC(y, y_pred_prob, model_type)

    
def superconductor_critical_temp_SVR():
    seed = 1
    X,y,scaler = load_data_superconductor(chem_data=True,formula_data=False,scaler=None,y_data="label",seed=1)
    param_grid = {
        "C": [100, 1000, 10000, 100000],
        "epsilon": [0.1, 1, 5, 10],
        "kernel": ["rbf"],
    }

    # Fitting 5 folds for each of 16 candidates, totalling 80 fits
    # duration = 8305.879885911942
    # Best Hyperparameters: {"C": 1000, "epsilon": 0.1, "kernel": "rbf"}
    # Final Model Performance on Full Data:
    # Mean Squared Error: 1.0688
    # R^2 Score: 0.9095
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    svr = SVR()
    grid_search = GridSearchCV(estimator=svr,
                               param_grid=param_grid,
                               cv=cv,
                               scoring="neg_mean_squared_error",
                               n_jobs=-1,
                               verbose=1)
    # grid_search.fit(X, y)
    # best_params = grid_search.best_params_
    # print("Best Hyperparameters:", best_params)
    best_params = {"C": 1000, "epsilon": 0.1, "kernel": "rbf"}
    best_svr = SVR(**best_params)
    best_svr.fit(X,y)
    y_pred = best_svr.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"Final Model Performance on Full Data:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

def tc_histgram():
    tc_ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
                 (50, 60), (60, 70), (70, 80), (80, 90), (90, 100),
                 (100, 110), (110, 120), (120, 130), (130, 140),
                 (140, 150), (150, 160), (160, 170), (170, 180), (180, 190)]
    counts = [6158, 2277, 1486, 1299, 687,
              637, 640, 981, 1256, 938,
              252, 199, 113, 84,
              2, 0, 0, 0, 1]
    data = []
    for (start, end), count in zip(tc_ranges, counts):
        midpoint = (start + end) / 2
        data.extend([midpoint] * count)
    data = np.array(data)
    bin_edges = [start for start, _ in tc_ranges] + [190]  # Add last edge
    plt.figure(figsize=(10, 6))
    sns.histplot(data, bins=bin_edges, edgecolor="black")
    plt.xlabel("Tc")
    plt.ylabel("Count")
    plt.title("Histogram of Tc Ranges")
    plt.xticks(np.arange(0, 200, 10))
    plt.tight_layout()
    plt.show()

def tc_histgram_f1score(data_type):
    tc_ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
                 (50, 60), (60, 70), (70, 80), (80, 90), (90, 100),
                 (100, 110), (110, 120), (120, 130), (130, 140),
                 (140, 150), (150, 160), (160, 170), (170, 180), (180, 190)]
    counts = [6158, 2277, 1486, 1299, 687, 637, 640, 981, 1256, 938,
              252, 199, 113, 84, 2, 0, 0, 0, 1]
    if data_type == "SVM":
        f1_a = [0.98, 0.87, 0.76, 0.78, 0.70, 0.69, 0.67, 0.72, 0.71, 0.77,
                0.86, 0.83, 0.82, 0.84, 0.94, 0.00, 0.00, 0.00, 0.92]
        f1_b = [0.97, 0.87, 0.77, 0.79, 0.72, 0.70, 0.69, 0.74, 0.73, 0.78,
                0.86, 0.83, 0.82, 0.83, 0.94, 0.00, 0.00, 0.00, 0.93]
    elif data_type == "DT":
        f1_a = [0.97,0.88,0.83,0.82,0.76,0.73,0.72,0.80,0.78,0.80,0.80,0.75,
                0.77,0.79,0.94,0.00,0.00,0.00,0.93]
        f1_b = [0.99,0.94,0.90,0.90,0.87,0.86,0.84,0.88,0.84,0.87,0.91,0.86,
                0.86,0.88,0.94,0.00,0.00,0.00,0.93]
    else:
        print("wrong data_type")

    data = []
    for (start, end), count in zip(tc_ranges, counts):
        midpoint = (start + end) / 2
        data.extend([midpoint] * count)
    data = np.array(data)
    bin_edges = [start for start, _ in tc_ranges] + [190]
    bin_midpoints = [(start + end) / 2 for start, end in tc_ranges]
    
    f1_a_filtered = [(x, y) for x, y in zip(bin_midpoints, f1_a) if y > 0]
    f1_b_filtered = [(x, y) for x, y in zip(bin_midpoints, f1_b) if y > 0]
    f1_a_x, f1_a_y = zip(*f1_a_filtered)
    f1_b_x, f1_b_y = zip(*f1_b_filtered)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.histplot(data, bins=bin_edges, edgecolor="black", ax=ax1, color="skyblue")
    ax1.set_xlabel("Tc")
    ax1.set_ylabel("Count", color="skyblue")
    ax1.set_title(f"Histogram of Tc Ranges with F1-score Chem Data and Chem+Formula Data for {data_type}")
    ax1.tick_params(axis="y", labelcolor="skyblue")
    ax1.set_xticks(np.arange(0, 200, 10))
    ax2 = ax1.twinx()
    ax2.plot(f1_a_x, f1_a_y, marker="o", linestyle="-", color="red", label="F1-score Chem Data")
    ax2.plot(f1_b_x, f1_b_y, marker="s", linestyle="--", color="green", label="F1-score Chem+Formula Data")
    ax2.set_ylabel("F1-score", color="black")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.legend(loc="center right")
    plt.tight_layout()
    plt.show()

def tc_histgram_f1score_both(name2):
    tc_ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
                 (50, 60), (60, 70), (70, 80), (80, 90), (90, 100),
                 (100, 110), (110, 120), (120, 130), (130, 140),
                 (140, 150), (150, 160), (160, 170), (170, 180), (180, 190)]
    counts = [6158, 2277, 1486, 1299, 687, 637, 640, 981, 1256, 938, 252, 199, 113, 84, 2, 0, 0, 0, 1]
    # SVM
    f1_a = [0.98, 0.87, 0.76, 0.78, 0.70, 0.69, 0.67, 0.72, 0.71, 0.77,
            0.86, 0.83, 0.82, 0.84, 0.94, 0.00, 0.00, 0.00, 0.92]
    f1_b = [0.97, 0.87, 0.77, 0.79, 0.72, 0.70, 0.69, 0.74, 0.73, 0.78,
            0.86, 0.83, 0.82, 0.83, 0.94, 0.00, 0.00, 0.00, 0.93]
    # Decision Tree
    if name2 == "DT":
        f1_c = [0.97,0.88,0.83,0.82,0.76,0.73,0.72,0.80,0.78,0.80,0.80,0.75,0.77,0.79,0.94,0.00,0.00,0.00,0.93]
        f1_d = [0.99,0.94,0.90,0.90,0.87,0.86,0.84,0.88,0.84,0.87,0.91,0.86,0.86,0.88,0.94,0.00,0.00,0.00,0.93]

    # Cost Sensitive
    if name2 == "CSSVM":
        f1_c = [0.97,0.85,0.74,0.77,0.68,0.67,0.67,0.72,0.71,0.77,0.86,0.83,0.82,0.84,0.94,0.00,0.00,0.00,0.93]
        f1_d = [0.97,0.85,0.75,0.78,0.70,0.69,0.69,0.75,0.73,0.78,0.87,0.82,0.82,0.84,0.94,0.00,0.00,0.00,0.93]

    if name2 == "FSVM":
        f1_c = [0.97,0.86,0.74,0.76,0.67,0.65,0.64,0.71,0.70,0.75,0.85,0.82,0.82,0.83,0.94,0.00,0.00,0.00,0.93]
        f1_d = [0.97,0.87,0.77,0.78,0.72,0.70,0.68,0.74,0.73,0.78,0.86,0.83,0.82,0.83,0.94,0.00,0.00,0.00,0.93]

    
    bin_edges = [start for start, _ in tc_ranges] + [190]
    bin_midpoints = [(start + end) / 2 for start, end in tc_ranges]
    data = []
    for (start, end), count in zip(tc_ranges, counts):
        midpoint = (start + end) / 2
        data.extend([midpoint] * count)
    data = np.array(data)
    # Filter zero F1 scores
    def filter_f1(f1_list):
        return [(x, y) for x, y in zip(bin_midpoints, f1_list) if y > 0]
    f1_a_x, f1_a_y = zip(*filter_f1(f1_a))
    f1_b_x, f1_b_y = zip(*filter_f1(f1_b))
    f1_c_x, f1_c_y = zip(*filter_f1(f1_c))
    f1_d_x, f1_d_y = zip(*filter_f1(f1_d))
    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.histplot(data, bins=bin_edges, edgecolor="black", ax=ax1, color="skyblue")
    ax1.set_xlabel("Tc")
    ax1.set_ylabel("Count", color="skyblue")
    ax1.set_title(f"Histogram of Tc Ranges with F1-scores for SVM and {name2}")
    ax1.tick_params(axis="y", labelcolor="skyblue")
    ax1.set_xticks(np.arange(0, 200, 10))
    ax2 = ax1.twinx()
    ax2.plot(f1_a_x, f1_a_y, marker="o", linestyle="-", color="red", label="F1-score SVM Chem")
    ax2.plot(f1_b_x, f1_b_y, marker="s", linestyle="--", color="green", label="F1-score SVM Chem+Formula")
    ax2.plot(f1_c_x, f1_c_y, marker="^", linestyle="-.", color="blue", label=f"F1-score {name2} Chem")
    ax2.plot(f1_d_x, f1_d_y, marker="d", linestyle=":", color="purple", label=f"F1-score {name2} Chem+Formula")
    ax2.set_ylabel("F1-score", color="black")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.legend(loc="center right")
    plt.tight_layout()
    plt.show()

def superconductor_EDA():
    seed = 1
    filename = f"{DATA_PATH}/critical-temperature-of-superconductors/train.csv"
    df = pd.read_csv(filename)

    fig, axes = plt.subplots(21, 4, figsize=(20, 100))
    axes = axes.flatten()
    for i, column in enumerate(df.columns):
        sns.histplot(df[column], ax=axes[i], kde=False)
        axes[i].set_title(column)
    for j in range(len(df.columns), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.show()

class CostSensitiveSVC(BaseEstimator, ClassifierMixin):
    def __init__(self, probability=False, **svc_kwargs):
        self.svc_kwargs = svc_kwargs
        self.probability = probability
        self.svc_kwargs["probability"] = probability
        self.model = SVC(**self.svc_kwargs)

    def fit(self, X, y):
        unique_classes = np.unique(y)
        class_weights_array = class_weight.compute_class_weight("balanced", classes=unique_classes, y=y)
        class_weight_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights_array)}
        sample_weights = np.array([class_weight_dict[c] for c in y])
        self.model = SVC(**self.svc_kwargs)
        self.model.fit(X, y, sample_weight=sample_weights)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self, deep=True):
        return self.svc_kwargs

    def set_params(self, **params):
        self.svc_kwargs.update(params)
        self.model = SVC(**self.svc_kwargs)
        return self

class FuzzySVC(BaseEstimator, ClassifierMixin):
    def __init__(self, epsilon=1e-3, probability=False, **svc_kwargs):
        self.epsilon = epsilon
        self.probability = probability
        self.svc_kwargs = svc_kwargs
        self.svc_kwargs["probability"] = probability
        self.model = SVC(**self.svc_kwargs)
        self.class_centers_ = None
        self.scaler_ = MinMaxScaler()

    def _compute_class_centers(self, X, y):
        class_centers = {}
        for cls in np.unique(y):
            class_centers[cls] = X[y == cls].mean(axis=0)
        return class_centers

    def _compute_fuzzy_memberships(self, X, y):
        distances = np.array([np.linalg.norm(x - self.class_centers_[label]) for x, label in zip(X, y)])
        normalized_distances = self.scaler_.fit_transform(distances.reshape(-1, 1)).flatten()
        memberships = 1 - normalized_distances
        memberships = np.clip(memberships, self.epsilon, 1 - self.epsilon)
        return memberships

    def fit(self, X, y):
        self.class_centers_ = self._compute_class_centers(X, y)
        sample_weights = self._compute_fuzzy_memberships(X, y)
        self.model = SVC(**self.svc_kwargs)
        self.model.fit(X, y, sample_weight=sample_weights)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def decision_function(self, X):
        return self.model.decision_function(X)

    def get_fuzzy_memberships(self, X, y):
        return self._compute_fuzzy_memberships(X, y)

    def get_params(self, deep=True):
        params = {"epsilon": self.epsilon}
        params.update(self.svc_kwargs)
        return params

    def set_params(self, **params):
        if "epsilon" in params:
            self.epsilon = params.pop("epsilon")
        self.svc_kwargs.update(params)
        self.model = SVC(**self.svc_kwargs)
        return self

def draw_ROC(y, y_pred, name=""):
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18]
    n_classes = len(classes)
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    y_indexed = np.array([class_to_index[label] for label in y])
    y_bin = label_binarize(y_indexed, classes=np.arange(n_classes))
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
        
def main():
    superconductor_EDA()
    superconductor_critical_temp_SVR()
    superconductor_critical_temp_SVC("SVC")
    superconductor_critical_temp_SVC("CSSVC")
    superconductor_critical_temp_SVC("FSVC")
    superconductor_critical_temp_dtc()
    tc_histgram()
    tc_histgram_f1score("SVM")
    tc_histgram_f1score("DT")
    tc_histgram_f1score_both("DT")
    tc_histgram_f1score_both("CSSVM")
    tc_histgram_f1score_both("FSVM")
    
main()
