import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict,Counter
import re
import scipy.stats as stats

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from dtreeviz.trees import *
import umap
from matplotlib.ticker import PercentFormatter
import matplotlib.patches as mpatches
import json

# Data: 2020 Annual Social and Economic Supplements
# URL: https://www.census.gov/data/datasets/2020/demo/cps/cps-asec-2020.html
# File: hhpub20.csv (included in asecpub20csv.zip)


def multiple_countplots(df,columns,vn=4,hn=3):
    f = lambda n:round(float(n),2)
    fig,ax = plt.subplots(vn,hn,figsize=(10,15),sharex=False, sharey=True)    
    for i,c in enumerate(columns):
        if c == "":
            ax[i//hn,i%hn].set_visible(False)
            continue
        sns.countplot(df,x=c,ax=ax[i//hn,i%hn])
        new_label = []
        for t in ax[i//hn,i%hn].get_xticklabels():
            txt = f(t.get_text())
            loc = t.get_position()
            new_label.append(plt.Text(position=loc,text=txt))
        ax[i//hn,i%hn].set_xticklabels(new_label)
        ax[i//hn,i%hn].grid()
    plt.show()
    plt.clf()

def multiple_histograms(df,columns,vn=4,hn=3):
    fig,ax = plt.subplots(vn,hn,figsize=(10,15),sharey=True)
    for i,c in enumerate(columns):
        if c == "":
            ax[i//hn,i%hn].set_visible(False)
            continue
        sns.histplot(df,x=c,bins=8,ax=ax[i//hn,i%hn])
        ax[i//hn,i%hn].set_yscale("log")
        ax[i//hn,i%hn].grid()
        ax[i//hn,i%hn].axvline(x=np.mean(df[c].tolist()),c="red")
    plt.show()
    plt.clf()

def multiple_boxplots(df,columns,vn=9,hn=2):
    fig,ax = plt.subplots(vn,hn,figsize=(10,15),sharex=True,sharey=True,squeeze=False)
    if len(columns) < vn*hn:
        columns += ["" for _ in range(vn*hn - len(columns))]
    for i,c in enumerate(columns):
        if c == "":
            ax[i//hn,i%hn].set_visible(False)
            continue
        sns.boxplot(df,x=c,ax=ax[i//hn,i%hn])
        ax[i//hn,i%hn].axvline(x=np.median(df[c].tolist()),c="red")
        ax[i//hn,i%hn].xaxis.label.set_size(8)
    plt.show()
    plt.clf()

def variable2desc(column):
    variable_dic = {"HHINC":"Total household income - recode",
                    "HPCTCUT":"Recode - HHLD income percentiles",
                    "HTOP5PCT":"Top 5 percent of households",
                    "HTOTVAL":"total household income",
                    "HEARNVAL":"total household earnings",
                    "HFRVAL":"household income - farm income",
                    "HINC_FR":"farm self-employment, y/n",
                    "HINC_SE":"own business self-employment, y/n",
                    "HINC_WS":"wage and salary, y/n",
                    "HSEVAL":"household income - self employment income",
                    "HWSVAL":"household income - wages and salaries",
                    "HANN_YN":"During 20.., did anyone receive income from an annuity?",
                    "HANNVAL":"household income - annuities",
                    "HCSP_YN":"During 20.. did anyone in this household receive: any child support payments?",
                    "HCSPVAL":"household income - child support",
                    "HDIS_YN":"Does anyone in the household have a disability or health problem which prevented them from working, even for a short time, or which limited the work they could do?",
                    "HDISVAL":"household income - disability income",
                    "HDIV_YN":"At any time during 20.. did anyone in this household: own any shares of stock in corporations or any mutual fund shares?",
                    "HDIVVAL":"household income - dividend income",
                    "HDST_YN":"Household retirement distribution income for people age 58 and over, y/n?",
                    "HDSTVAL":"household income - retirement distributions",
                    "HED_YN":"Did anyone receive any educational assistance for tuition, fees, books, or living expenses during 20..?",
                    "HEDVAL":"household income - education income",
                    "HFIN_YN":"During 20.. did anyone in this household receive: any (other) regular financial assistance from friends or relatives not living in this household?",
                    "HFINVAL":"household income - financial assistance income",
                    "HINC_UC":"unemployment compensation, y/n",
                    "HINC_WC":"workers compensation, y/n",
                    "HINT_YN":"At any time during 20.. did anyone in this household have money in: 1) savings accounts 2) checking accounts 3) money market funds 4) certificates of deposit 5) savings bonds 6) any other (non-retirement) investments which pay interest 7) retirement accounts",
                    "HINTVAL":"household income - interest income",
                    "HOI_YN":"During 20.. Did anyone receive cash income not already covered, such as income from: foster child care, alimony, jury duty, armed forces reserves, severance pay, hobbies, or any other source?",
                    "HOIVAL":"household income - other income: (such as foster child care, alimony, jury duty, armed forces reserves, severance pay, hobbies, or any other source)",
                    "HOTHVAL":"All other types of income except HEARNVAL Recode - Total other household income",
                    "HPAW_YN":"At any time during 20.. did anyone in this household receive: any public assistance or welfare payments from the state or local welfare office?",
                    "HPAWVAL":"household income - public assistance income amt",
                    "HPEN_YN":"During 20.., did anyone receive any pension income from a previous employer or union?",
                    "HPENVAL":"household income - pension income",
                    "HRNT_YN":"During 20.. did anyone in the household: 1) own any land, business property, apartments, houses which were rented to others? 2) receive income from royalties or from roomers or boarders? 3) receive income from estates or trusts?",
                    "HRNTVAL":"household income - rental income amt",
                    "HSS_YN":"During 20.. did anyone in this household receive: any social security payments from U.S. government?",
                    "HSSI_YN":"During 20.. did anyone in this household receive: any supplemental security income payments?",
                    "HSSIVAL":"household income - supplemental security income",
                    "HSSVAL":"household income - social security",
                    "HSUR_YN":"Did anyone in this household receive any income in 20.. as a survivor or widow such as survivor or widow's pensions, estates, trusts, annuities, or other survivor benefits?",
                    "HSURVAL":"household income - survivor income",
                    "HUCVAL":"household income - unemployment compensation",
                    "HVET_YN":"At any time during 20.. did anyone in this household receive: any payments from the veterans' administration other than above?",
                    "HVETVAL":"household income - veteran payments",
                    "HWCVAL":"household income - worker's compensation",
                    "HENGAST":"Assistance for heating/colling costs received for anyone in the household",
                    "HENGVAL":"Altogether, how much energy assistance has been received during, 20..?",
                    "HFDVAL":"What was the value of all food stamps received during 20..?",
                    "HFLUNCH":"During 20.. how many of the children in this household received free or reduced price lunches because they qualified for federal school lunch program?",
                    "HFLUNNO":"Number receiving free/reduced price lunch. Note: if more than 9 children/persons present, a value of 9 does not necessarily mean all.",
                    "HFOODMO":"number months covered by food stamps",
                    "HFOODNO":"Number covered by food stamps note: if more than 9 children/persons present, a value of 9 does not necessarily mean all.",
                    "HFOODSP":"Did anyone in this household get food stamps at any time in 20..?",
                    "HHOTLUN":"During 20.. how many of the children in this household usually ate a complete hot lunch offered at school?",
                    "HHOTNO":"number of children in household who usually ate hot lunch. note: if more than 9 children/persons present, a value of 9 does not necessarily mean all.",
                    "HLORENT":"Are you paying lower rent because the federal, state, or local government is paying part of the cost?",
                    "HPUBLIC":"Is this a public housing project, that is owned by a local housing authority or other public agency?",
                    "HRNUMWIC":"Number of people in the household receiving WIC",
                    "HRWICYN":"At any time last year, (were you/was anyone in this household) on WIC, the Women, Infants, and Children Nutrition Program?",
                    "HCHCARE_VAL":"Annual amount paid for child care by household members",
                    "HCHCARE_YN":"Did (you/anyone in this household) PAY for the care of (your/their) (child/children) while they worked last year? (Include preschool and nursery school; exclude kindergarten or grade/elementary school)?",
                    "HPRES_MORT":"Presence of home mortgage (respondent answers yes to hmort_yn or hsmort_yn)",
                    "HPROP_VAL":"Estimate of current property value",
                    "I_CHCAREVAL":"Allocation flag for HCHCARE_VAL",
                    "I_HENGAS":"Allocation flag for HENGAST",
                    "I_HENGVA":"Allocation flag for HENGVAL",
                    "I_HFDVAL":"Allocation flag for HFDVAL",
                    "I_HFLUNC":"Allocation flag for HFLUNCH",
                    "I_HFLUNN":"Allocation flag for HFLUNNO",
                    "I_HFOODM":"Allocation flag for HFOODMO",
                    "I_HFOODN":"Allocation flag for HFOODNO",
                    "I_HFOODS":"Allocation flag for HFOODSP",
                    "I_HHOTLU":"Allocation flag for HHOTLUN",
                    "I_HHOTNO":"Allocation flag for HHOTNO",
                    "I_HLOREN":"Allocation flag for HLORENT",
                    "I_HPUBLI":"Allocation flag for HPUBLIC",
                    "I_PROPVAL":"Allocation flag for HPROP_VAL",
                    "THCHCARE_VAL":"Topcode flag for HCHCARE_VAL",
                    "THPROP_VAL":"Data swapping flag for HPROP_VAL",
                    "HCOV":"Any health insurance coverage in the household last year",
                    "NOW_HCOV":"Any current health insurance coverage in the household",
                    "HPUB":"Any public coverage in the household last year",
                    "NOW_HPUB":"Any current public coverage in the household",
                    "HPRIV":"Any private coverage in the household last year",
                    "NOW_HPRIV":"Any current private coverage in the household",
                    "HMCAID":"Any Medicaid, PCHIP or other means-tested coverage in the household last year",
                    "NOW_HMCAID":"Any current Medicaid, PCHIP or other means-tested coverage in the household",
                    "HH_HI_UNIV":"Household imputation status"}
    if column in variable_dic:
        return variable_dic[column]
    else:
        return ""
    
def print_df(df, cols):
    '''
    Print variable names, description, and number of unique values
    '''
    uniqs = dict()
    for c in cols:
        uniqs[c] = int(df[c].nunique())
    cols = list(uniqs.keys())
    qtns = list(map(lambda c:variable2desc(c)[:20], cols))
    unqs = list(uniqs.values())
    print(pd.DataFrame(zip(cols, qtns, unqs),
                       columns=["Column","Description","Unique value"]))

def print_skew_maxvalue(df, columns):
    '''
    Print variable names, skewness and max value
    '''
    skews = dict()
    maxvals = dict()
    for c in columns:
        skews[c] = stats.skew(df[c].tolist())
        maxvals[c] = max(df[c].tolist())
    skews = {k:v for k,v in sorted(skews.items(), key=lambda itm:itm[1], reverse=True)}
    df_skews = pd.DataFrame(skews.keys(),columns=["Column Name"])
    df_skews["Skew"] = skews.values()
    df_skews["Max Value"] = [maxvals[c] for c in skews.keys()]
    print(df_skews)

def check_correlations(df):
    '''
    Creates a heatmap for the given dataframe
    '''
    covm = np.cov(df.values, rowvar=False, bias=True)
    plt.title("Covariance matrix between variables")
    sns.heatmap(covm, cmap="YlGnBu", vmin=-1, vmax=1)
    plt.show()
    plt.clf()

def create_pareto_chart(y):
    '''
    Create a Pareto Chart for explained variances
    the left side y-axis is for explained variances and the right side y-axis is for its percentage
    '''
    x = np.arange(len(y))
    y2 = []
    total = 0
    for val in y / sum(y) * 100:
        total += val
        y2.append(total)

    fig, ax = plt.subplots()
    ax.bar(x, y,color="C0")
    ax2 = ax.twinx()
    ax2.plot(x, y2, "o-", color="C1")
    ax2.yaxis.set_major_formatter(PercentFormatter())

    ax.tick_params(axis="y", colors="C0")
    ax2.tick_params(axis="y", colors="C1")
    plt.title("Explained Variances from PCA")
    plt.show()

# a global variable to store a dimension reduction result by UMAP
df_reduced_saved = None
def show_clustering(df, n_clusters_list):
    global df_reduced_saved
    df_reduced = None
    if df_reduced_saved is not None:
        df_reduced = df_reduced_saved
    else:
        reducer = umap.UMAP(n_neighbors=100, n_components=2)
        df_reduced = pd.DataFrame(reducer.fit_transform(df), columns=["x","y"])
        df_reduced_saved = df_reduced
        
    colors = ["blue","orange","green","red","purple",
              "brown","pink","gray","olive","cyan",
              "lime","navy","gold"]
    fig,ax = plt.subplots(4,2,figsize=(15,15))
    for i,n_clusters in enumerate(n_clusters_list):
        model = KMeans(n_clusters=n_clusters, n_init=10)
        preds = model.fit_predict(df.values)
        groups = [str(j) for j in range(n_clusters)]
        pred_colors = list(map(lambda p:colors[p], preds))
        ax[i,0].scatter(df_reduced["x"], df_reduced["y"], c=pred_colors)
        recs = []
        for j in range(n_clusters):
            recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[j]))
        ax[i,0].legend(recs, groups, loc=4)

        df_preds = pd.DataFrame(preds,columns=["Group"])
        sns.countplot(x=df_preds["Group"], palette=colors[:n_clusters], ax=ax[i,1])

    fig.suptitle(f"K-Mean clustering to {n_clusters_list[0]} ~ {n_clusters_list[-1]} groups on 2D by UMAP")
    plt.show()
    plt.clf()

def show_clustering_detail(df, df2, n_clusters):
    '''
    Display the colored clustering result in the left side (ax[0])
    along with the mean values of variables in the right side (ax[1])
    '''
    global df_reduced_saved
    df_reduced = None
    if df_reduced_saved is not None:
        df_reduced = df_reduced_saved
    else:
        reducer = umap.UMAP(n_neighbors=100, n_components=2)
        df_reduced = pd.DataFrame(reducer.fit_transform(df), columns=["x","y"])
        df_reduced_saved = df_reduced

    model = KMeans(n_clusters=n_clusters, n_init=10)
    df2["Group"] = model.fit_predict(df.values)
    v = []
    for c in df.columns:
        row = [c]
        for i in range(n_clusters):
            row.append(round(df2[df2["Group"]==i][c].mean(),2))
        v.append(row)
    v.append(["Data Number"] + [len(df2[df2["Group"]==i]) for i in range(n_clusters)])

    colors = ["blue","orange","green","red","purple",
              "brown","pink","gray","olive","cyan",
              "lime","navy","gold"]
    fig,ax = plt.subplots(1,2, figsize=(15,15))
    groups = [str(j) for j in range(n_clusters)]
    pred_colors = list(map(lambda p:colors[p], df2["Group"].tolist()))
    ax[0].scatter(df_reduced["x"], df_reduced["y"], c=pred_colors)
    recs = []
    for j in range(n_clusters):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[j]))
    ax[0].legend(recs, groups, loc=4)

    df_data = pd.DataFrame(np.array(v).reshape(len(df.columns)+1, n_clusters+1),
                           columns=["Variable"]+list(range(n_clusters)))
    ax[1].table(cellText=df_data.values, colLabels=df_data.columns, loc='center')
    fig.patch.set_visible(False)
    ax[1].axis('off')
    ax[1].axis('tight')
    fig.tight_layout()
    plt.show()
    plt.clf()

def topic2columns(topic_num):
    if topic_num == 6:
        return ['HHINC', 'HPCTCUT', 'HTOP5PCT', 'HTOTVAL', 'HEARNVAL',
                'HFRVAL', 'HINC_FR', 'HINC_SE', 'HINC_WS', 'HSEVAL',
                'HWSVAL', 'HANN_YN', 'HANNVAL', 'HCSP_YN', 'HCSPVAL',
                'HDIS_YN', 'HDISVAL', 'HDIV_YN', 'HDIVVAL', 'HDST_YN',
                'HDSTVAL', 'HED_YN', 'HEDVAL', 'HFIN_YN', 'HFINVAL',
                'HINC_UC', 'HINC_WC', 'HINT_YN', 'HINTVAL', 'HOI_YN',
                'HOIVAL', 'HOTHVAL', 'HPAW_YN', 'HPAWVAL', 'HPEN_YN',
                'HPENVAL', 'HRNT_YN', 'HRNTVAL', 'HSS_YN', 'HSSI_YN',
                'HSSIVAL', 'HSSVAL', 'HSUR_YN', 'HSURVAL', 'HUCVAL',
                'HVET_YN', 'HVETVAL', 'HWCVAL', 'HENGAST', 'HENGVAL',
                'HFDVAL', 'HFLUNCH', 'HFLUNNO', 'HFOODMO', 'HFOODNO',
                'HFOODSP', 'HHOTLUN', 'HHOTNO', 'HLORENT', 'HPUBLIC',
                'HRNUMWIC', 'HRWICYN', 'HCHCARE_VAL', 'HCHCARE_YN',
                'HPRES_MORT', 'HPROP_VAL', 'I_CHCAREVAL', 'I_HENGAS',
                'I_HENGVA', 'I_HFDVAL', 'I_HFLUNC', 'I_HFLUNN',
                'I_HFOODM', 'I_HFOODN', 'I_HFOODS', 'I_HHOTLU',
                'I_HHOTNO', 'I_HLOREN', 'I_HPUBLI', 'I_PROPVAL',
                'THCHCARE_VAL', 'THPROP_VAL']
    if topic_num == 7:
        return ['HCOV', 'NOW_HCOV', 'HPUB', 'NOW_HPUB', 'HPRIV',
                'NOW_HPRIV', 'HMCAID', 'NOW_HMCAID', 'HH_HI_UNIV']
    return []

def check_imcome(df):
    '''
    Conduct EDA for Income topic variables
    '''
    # Pull out the Income topic variable names from document
    # 6 for Income topic
    # 7 for Health Insurance topic
    income_columns = topic2columns(6)

    # Remove re-coding variables.
    for c in ["HHINC","HPCTCUT","HTOP5PCT"]:
        income_columns.remove(c)

    # Remove all the "Flag" variables
    for c in ["I_CHCAREVAL","I_HENGAS","I_HENGVA","I_HFDVAL",
              "I_HFLUNC","I_HFLUNN","I_HFOODM","I_HFOODN",
              "I_HFOODS","I_HHOTLU","I_HHOTNO","I_HLOREN",
              "I_HPUBLI","I_PROPVAL","THCHCARE_VAL","THPROP_VAL"]:
        income_columns.remove(c)

    # data cleaning for binary variables
    # 30 binary variables
    yn_cls1 = ["HANN_YN","HCSP_YN","HDIS_YN",
               "HDIV_YN","HDST_YN","HED_YN",
               "HFIN_YN","HINT_YN","HOI_YN",
               "HPAW_YN","HPEN_YN","HRNT_YN"]
    yn_cls2 = ["HSS_YN","HSSI_YN","HSUR_YN",
               "HVET_YN","HRWICYN","HCHCARE_YN",
               "HINC_FR", "HINC_SE","HINC_WS",
               "HINC_UC", "HINC_WC","HENGAST"]
    yn_cls3 = ["HFLUNCH", "HFOODSP","HHOTLUN",
               "HLORENT", "HPUBLIC","HPRES_MORT",
               "", "", "",
               "", "", ""]
    yn_cls3a = ["HFLUNCH", "HFOODSP","HHOTLUN",
                "HLORENT", "HPUBLIC","HPRES_MORT"]

    # Convert the output format of binary variables
    # NIU: 0 ==> 0
    # Yes: 1 ==> 1
    # No:  2 ==> -1
    for c in yn_cls1 + yn_cls2 + yn_cls3:
        if c != "":
            df[c].replace([0,1,2], [0,1,-1], inplace=True)

    # Cut out the Income topic variables from the entire data frame
    df = df[income_columns]
    # Before standardization, keep a copy of the original values for later use
    df_orig_value = df.copy()
    
    # Standerdize the Income topic variables (63 variables x 91,500 observations)
    # To skip standardization, comment out the following two lines
    transformer = StandardScaler().fit(df.values)
    df = pd.DataFrame(transformer.transform(df.values), columns=df.columns)

    # Create histograms for categorical variables
    multiple_countplots(df,yn_cls1)
    multiple_countplots(df,yn_cls2)
    multiple_countplots(df,yn_cls3)
    
    # Now data clearning discrete/continuous variables
    # 33 discrete/continuous variables
    val_cls1 = ["HTOTVAL", "HEARNVAL", "HFRVAL", 
                "HSEVAL", "HWSVAL", "HANNVAL", 
                "HCSPVAL", "HDISVAL", "HDIVVAL", 
                "HDSTVAL", "HEDVAL", "HFINVAL"]
    val_cls2 = ["HINTVAL", "HOIVAL", "HOTHVAL", 
                "HPAWVAL", "HPENVAL", "HRNTVAL", 
                "HSSIVAL", "HSSVAL", "HSURVAL", 
                "HUCVAL", "HVETVAL", "HWCVAL"]
    val_cls3 = ["HENGVAL", "HFDVAL", "HFLUNNO", 
                "HFOODMO", "HFOODNO", "HHOTNO", 
                "HRNUMWIC", "HCHCARE_VAL", "HPROP_VAL"]

    # Create histograms for discrete/continuous variables
    multiple_histograms(df,val_cls1)
    multiple_histograms(df,val_cls2)
    multiple_histograms(df,val_cls3,vn=3)

    # Print number of unique values 
    print_df(df,val_cls1)
    print_df(df,val_cls2)
    print_df(df,val_cls3)
    
    # Calculate the covariance matrix for Income topic variables and show the result in Heatmap
    # between categorical variables
    check_correlations(df[yn_cls1  + yn_cls2  + yn_cls3a])
    # between discrete/continuous variables
    check_correlations(df[val_cls1 + val_cls2 + val_cls3])
    # between all variables
    check_correlations(df[yn_cls1  + yn_cls2  + yn_cls3a + val_cls1 + val_cls2 + val_cls3])

    # Print variable names, its skewness and max values in the descending order
    print_skew_maxvalue(df, val_cls1+val_cls2+val_cls3)

    # Create boxplot for all the variables
    multiple_boxplots(df, df.columns[ 0:18].tolist())
    multiple_boxplots(df, df.columns[18:36].tolist())
    multiple_boxplots(df, df.columns[36:54].tolist())
    multiple_boxplots(df, df.columns[54:len(df.columns)].tolist(),vn=5,hn=2)

    # Create a Pareto chart for explained variance for PCA
    reducer = PCA()
    reducer.fit(df)
    create_pareto_chart(reducer.explained_variance_)

    # Enable this to do elbow method.
    if False:
        sses = dict()
        for k in range(2,30):
            t1 = time.time()
            model = KMeans(n_clusters=k,n_init=10)
            preds = model.fit_predict(df_result.values)
            sses[k] = model.inertia_
            t2 = time.time()
            print(f"done {k}", round(t2-t1))

        x = sses.keys()
        y = sses.values()
        plt.title("Elbow method no PCA")
        plt.xlabel("Cluster size")
        plt.ylabel("Sum of Square Errors")
        plt.plot(x,y,"o-")
        plt.grid()
        plt.show()
        plt.clf()
        
    # Run K-Means and show result in color and frequency graph
    show_clustering(df, [2,3,4,5])
    show_clustering(df, [6,7,8,9])
    show_clustering(df, [10,11,12,13])

    # Run K-Means and show result in color and mean values
    show_clustering_detail(df,df_orig_value,2)
    show_clustering_detail(df,df_orig_value,4)
    show_clustering_detail(df,df_orig_value,7)

    return

def main():
    data_file = "hhpub20.csv"
    df = pd.read_csv(data_file)

    print(f"Number of observations: {len(df)}")
    print(f"Number of variables: {df.shape[1]}")
    
    check_imcome(df)

    print("done")
    return

main()
