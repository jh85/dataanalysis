import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
import scipy.io as sio
from scipy.signal import get_window
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



def plot_feature_importance(model, model_name, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Top 10 features
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(indices)), importances[indices], align="center")
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
    plt.title(f"{model_name} - Top 10 Feature Importances")
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def compare(X,y,feature_names):
    print(f"X={X.shape} y={y.shape}")
    seed = 1
    X_train2,X_test,y_train2,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=seed)

    X_train, y_train = SMOTE().fit_resample(X_train2, y_train2)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    my_metric = "accuracy"
    # my_metric = "f1"
    # Random Forest
    param_grid_rfc = {"n_estimators": [400, 800],
                      "max_depth": [None, 10, 20],
                      "min_samples_split": [2],
                      "min_samples_leaf": [1, 2],
                      "max_features": ["sqrt"],
                      "criterion": ["gini", "entropy", "log_loss"],
                      "random_state": [seed]}
    model_rfc = RandomForestClassifier(random_state=seed)
    grid_rfc = GridSearchCV(estimator=model_rfc,
                            param_grid=param_grid_rfc,
                            cv=cv,
                            scoring=my_metric,
                            n_jobs=-1,
                            verbose=0)
    grid_rfc.fit(X_train, y_train)
    best_model_rfc = grid_rfc.best_estimator_
    n_grids_rfc = len(grid_rfc.cv_results_["params"])
    fit_time_rfc = np.mean(grid_rfc.cv_results_["mean_fit_time"])
    score_time_rfc = np.mean(grid_rfc.cv_results_["mean_score_time"])

    # ExtraTrees
    param_grid_etc = {"n_estimators": [400, 800],
                      "max_depth": [None, 10, 20],
                      "min_samples_split": [2],
                      "min_samples_leaf": [1, 2],
                      "max_features": ["sqrt"],
                      "bootstrap": [False],  # additional param for ExtraTree
                      "criterion": ["gini", "entropy"],
                      "random_state": [seed]}
    model_etc = ExtraTreesClassifier(random_state=seed)
    grid_etc = GridSearchCV(estimator=model_etc,
                            param_grid=param_grid_etc,
                            cv=cv,
                            scoring=my_metric,
                            n_jobs=-1,
                            verbose=0)
    grid_etc.fit(X_train, y_train)
    best_model_etc = grid_etc.best_estimator_
    n_grids_etc = len(grid_etc.cv_results_["params"])
    fit_time_etc = np.mean(grid_etc.cv_results_["mean_fit_time"])
    score_time_etc = np.mean(grid_etc.cv_results_["mean_score_time"])

    # AdaBoost
    param_grid_ada = {"n_estimators": [400, 800],
                      "learning_rate": [0.5, 1.0, 1.5, 2.0],
                      "random_state": [seed]}
    model_ada = AdaBoostClassifier(random_state=seed)
    grid_ada = GridSearchCV(estimator=model_ada,
                            param_grid=param_grid_ada,
                            cv=cv,
                            scoring=my_metric,
                            n_jobs=-1,
                            verbose=0)
    grid_ada.fit(X_train, y_train)
    best_model_ada = grid_ada.best_estimator_
    n_grids_ada = len(grid_ada.cv_results_["params"])
    fit_time_ada = np.mean(grid_ada.cv_results_["mean_fit_time"])
    score_time_ada = np.mean(grid_ada.cv_results_["mean_score_time"])

    # XGBoost
    param_grid_xgb = {
        "n_estimators": [400, 800],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "use_label_encoder": [False],
        # "eval_metric": ["logloss"]
        "objective": ["binary:logitraw"],
        "random_state": [seed]
    }
    model_xgb = XGBClassifier(random_state=seed, verbosity=0)
    grid_xgb = GridSearchCV(estimator=model_xgb,
                            param_grid=param_grid_xgb,
                            cv=cv,
                            scoring=my_metric,
                            n_jobs=-1,
                            verbose=0)
    grid_xgb.fit(X_train, y_train)
    best_model_xgb = grid_xgb.best_estimator_
    n_grids_xgb = len(grid_xgb.cv_results_["params"])
    fit_time_xgb = np.mean(grid_xgb.cv_results_["mean_fit_time"])
    score_time_xgb = np.mean(grid_xgb.cv_results_["mean_score_time"])

    # LightGBM
    param_grid_lgb = {"n_estimators": [400, 800],
                      "max_depth": [3, 5, 7],
                      "learning_rate": [0.05, 0.1, 0.2],
                      "num_leaves": [31, 63],
                      "is_unbalance": [True],  # this setting handles class imbalance
                      "boosting_type": ["gbdt"],
                      "random_state": [seed]}
    model_lgb = LGBMClassifier(random_state=seed, verbose=-1)
    grid_lgb = GridSearchCV(estimator=model_lgb,
                            param_grid=param_grid_lgb,
                            scoring=my_metric,
                            cv=cv,
                            n_jobs=-1,
                            verbose=0)
    # to avoid "X does not have valid feature names,..." UserWarning
    df_X_train = pd.DataFrame(data=X_train, columns=[i for i in range(X_train.shape[1])])
    grid_lgb.fit(df_X_train, y_train)
    best_model_lgb = grid_lgb.best_estimator_
    n_grids_lgb = len(grid_lgb.cv_results_["params"])
    fit_time_lgb = np.mean(grid_lgb.cv_results_["mean_fit_time"])
    score_time_lgb = np.mean(grid_lgb.cv_results_["mean_score_time"])


    y_pred_rfc = best_model_rfc.predict(X_test)
    print("Random Forest")
    print("Best parameters:", grid_rfc.best_params_)
    print("Classification Report on Test Set:")
    print(classification_report(y_test, y_pred_rfc))
    print("Test Accuracy:", accuracy_score(y_test, y_pred_rfc))
    print("mean fit time:", fit_time_rfc)
    print("mean score time:", score_time_rfc)
    print("total combinations:", n_grids_rfc)
    
    y_pred_etc = best_model_etc.predict(X_test)
    print("ExtraTrees")
    print("Best parameters:", grid_etc.best_params_)
    print("Classification Report on Test Set:")
    print(classification_report(y_test, y_pred_etc))
    print("Test Accuracy:", accuracy_score(y_test, y_pred_etc))
    print("mean fit time:", fit_time_etc)
    print("mean score time:", score_time_etc)
    print("total combinations:", n_grids_etc)

    y_pred_ada = best_model_ada.predict(X_test)
    print("Adaboost")
    print("Best Parameters:", grid_ada.best_params_)
    print("Classification Report on Test Set:")
    print(classification_report(y_test, y_pred_ada))
    print("Test Accuracy:", accuracy_score(y_test, y_pred_ada))
    print("mean fit time:", fit_time_ada)
    print("mean score time:", score_time_ada)
    print("total combinations:", n_grids_ada)

    y_pred_xgb = best_model_xgb.predict(X_test)
    print("XGBoost")
    print("Best Parameters:", grid_xgb.best_params_)
    print("Classification Report on Test set:")
    print(classification_report(y_test, y_pred_xgb))
    print("Test Accuracy:", accuracy_score(y_test, y_pred_xgb))
    print("mean fit time:", fit_time_xgb)
    print("mean score time:", score_time_xgb)
    print("total combinations:", n_grids_xgb)

    y_pred_lgb = best_model_lgb.predict(X_test)
    print("LightGBM")
    print("Best Parameters:", grid_lgb.best_params_)
    print("Classification Report on Test set:")
    print(classification_report(y_test, y_pred_lgb))
    print("Test Accuracy:", accuracy_score(y_test, y_pred_lgb))
    print("mean fit time:", fit_time_lgb)
    print("mean score time:", score_time_lgb)
    print("total combinations:", n_grids_lgb)

    
    y_prob_rfc = best_model_rfc.predict_proba(X_test)[:, 1]
    fpr_rfc, tpr_rfc, _ = roc_curve(y_test, y_prob_rfc)
    roc_auc_rfc = auc(fpr_rfc, tpr_rfc)
    
    y_prob_etc = best_model_etc.predict_proba(X_test)[:, 1]
    fpr_etc, tpr_etc, _ = roc_curve(y_test, y_prob_etc)
    roc_auc_etc = auc(fpr_etc, tpr_etc)

    y_prob_ada = best_model_ada.predict_proba(X_test)[:, 1]
    fpr_ada, tpr_ada, _ = roc_curve(y_test, y_prob_ada)
    roc_auc_ada = auc(fpr_ada, tpr_ada)

    y_prob_xgb = best_model_xgb.predict_proba(X_test)[:, 1]
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
    roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

    y_prob_lgb = best_model_lgb.predict_proba(X_test)[:, 1]
    fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_prob_lgb)
    roc_auc_lgb = auc(fpr_lgb, tpr_lgb)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_rfc, tpr_rfc, label=f"RandomForest (AUC = {roc_auc_rfc:.2f})", linewidth=2)
    plt.plot(fpr_etc, tpr_etc, label=f"ExtraTrees (AUC = {roc_auc_etc:.2f})", linewidth=2)
    plt.plot(fpr_ada, tpr_ada, label=f"AdaBoost (AUC = {roc_auc_ada:.2f})", linewidth=2)
    plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {roc_auc_xgb:.2f})", linewidth=2)
    plt.plot(fpr_lgb, tpr_lgb, label=f"LightGBM (AUC = {roc_auc_lgb:.2f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    precision_rfc, recall_rfc, _ = precision_recall_curve(y_test, y_prob_rfc)
    ap_rfc = average_precision_score(y_test, y_prob_rfc)
    
    precision_etc, recall_etc, _ = precision_recall_curve(y_test, y_prob_etc)
    ap_etc = average_precision_score(y_test, y_prob_etc)

    precision_ada, recall_ada, _ = precision_recall_curve(y_test, y_prob_ada)
    ap_ada = average_precision_score(y_test, y_prob_ada)

    precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_prob_xgb)
    ap_xgb = average_precision_score(y_test, y_prob_xgb)

    precision_lgb, recall_lgb, _ = precision_recall_curve(y_test, y_prob_lgb)
    ap_lgb = average_precision_score(y_test, y_prob_lgb)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_rfc, precision_rfc, label=f"RandomForest (AP = {ap_rfc:.2f})", linewidth=2)
    plt.plot(recall_etc, precision_etc, label=f"ExtraTrees (AP = {ap_etc:.2f})", linewidth=2)
    plt.plot(recall_ada, precision_ada, label=f"AdaBoost (AP = {ap_ada:.2f})", linewidth=2)
    plt.plot(recall_xgb, precision_xgb, label=f"XGBoost (AP = {ap_xgb:.2f})", linewidth=2)
    plt.plot(recall_lgb, precision_lgb, label=f"LightGBM (AP = {ap_lgb:.2f})", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plot_feature_importance(best_model_rfc, "RandomForest", feature_names)
    plot_feature_importance(best_model_etc, "ExtraTrees",   feature_names)
    plot_feature_importance(best_model_xgb, "XGBoost",      feature_names)
    plot_feature_importance(best_model_lgb, "LightGBM",     feature_names)
    plot_feature_importance(best_model_ada, "AdaBoost",     feature_names)

    
    algorithms = ["RFC", "ETC", "Ada", "XGB", "LGB"]
    data = pd.DataFrame({
        "Algorithm": algorithms + algorithms,
        "Time (seconds)": [fit_time_rfc, fit_time_etc, fit_time_ada, fit_time_xgb, fit_time_lgb,
                          score_time_rfc, score_time_etc, score_time_ada, score_time_xgb, score_time_lgb],
        "Metric": ["Fit Time"] * 5 + ["Score Time"] * 5
    })
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Algorithm", y="Time (seconds)", hue="Metric", data=data, palette="viridis")
    plt.title("Comparison of Fit Time vs Score Time Across Algorithms", fontsize=15)
    plt.xlabel("Algorithm", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Metric", loc="upper right")
    for i, bar in enumerate(plt.gca().patches):
        plt.gca().text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.05,
            f"{bar.get_height():.2f}s",
            ha="center",
            fontsize=9
        )
    plt.tight_layout()
    plt.show()
    
def load_cervical_cancer_data():
    filename = "risk_factors_cervical_cancer.csv"
    df = pd.read_csv(filename)
    df = df.replace("?",np.nan)
    # 90% are missing
    df = df.drop(columns=["STDs: Time since first diagnosis","STDs: Time since last diagnosis"])
    df = df.apply(pd.to_numeric)
    df = df.fillna(df.mean())
    columns = list(df.columns)
    columns.remove("Biopsy")
    X = df[columns].values
    y = df["Biopsy"].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
 
    short_labels = ["Age", "SexPart", "FirstSex", "Pregs", "Smoke", "SmokeYrs", "SmokePks",
                    "HormCon", "HormYrs", "IUD", "IUDYrs", "STDs", "STDNum", "STDCond", "STDCerv", 
                    "STDVag", "STDVulv", "STDSyph", "STD_PID", "STDHerp", "STDMoll", "STDAIDS",
                    "STDHIV", "STDHepB", "STDHPV", "STDDiag", "DxCa", "DxCIN", "DxHPV", "Dx",
                    "Hinsel", "Schiller", "Cito"]
    return X,y,short_labels

def cervical_cancer():
    X,y,feature_names = load_cervical_cancer_data()
    compare(X,y,feature_names)

def mystft(signal, window_size, sr=1, step=1):
    window = get_window("hann", window_size)
    n_segments = (len(signal) - window_size) // step + 1
    segments = np.lib.stride_tricks.sliding_window_view(signal, window_shape=window_size)[::step]
    segments_zero_mean = segments - segments.mean(axis=1, keepdims=True)
    windowed = segments_zero_mean * window
    Zxx = np.fft.rfft(windowed, axis=1)
    power = np.abs(Zxx) ** 2
    f = np.fft.rfftfreq(window_size, d=1/sr)
    t = np.arange(n_segments) * step / sr
    return power.T, t, f

def load_anesthetized_data():
    x = data()
    step = 50
    window_size = 1000
    pwr,t,f = mystft(x, window_size, 1000, step)
    y = np.zeros(pwr.shape[1])
    anesthetized_start = int( 553.55 * 1000 / step) + 1
    anesthetized_end   = int(1098.10 * 1000 / step)
    y[anesthetized_start:anesthetized_end] = 1
    X = pwr.T[0:101:5]
    feature_names = [f"{i}Hz" for i in range(0,101,5)]
    return X,y,feature_names

def detect_anesthetized_state():
    X,y,feature_names = load_anesthetized_data()
    compare(X,y,feature_names)

def main():
    cervical_cancer()
    detect_anesthetized_state()
    
main()
