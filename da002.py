import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats
import math
from sklearn.linear_model import LinearRegression, HuberRegressor

# data: Retail Food Stores in New York State
# URL: https://data.ny.gov/Economic-Development/Retail-Food-Stores/9a8c-vfzj
# file: Retail_Food_Stores.csv

def show_frequency(df, column_name, top_n=10, drop_other=False, logscale=False,
                   rotation=0, filename=None, title=None, xlabel=None, ylabel=None):
    arr = df[column_name].tolist()
    freq = defaultdict(int)
    for i in range(len(arr)):
        freq[str(arr[i])] += 1
    freq_sorted = {k:v for k,v in sorted(freq.items(), key=lambda itm:itm[1], reverse=True)}

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
    plt.xticks(rotation=rotation, ha="right")
    plt.grid()
    if filename:
        plt.savefig("filename", bbox_inches="tight")
    plt.show()

def show_histogram(df, column_name, logscale=False, logscale_x=False,
                  rotation=0, filename=None, title=None, xlabel=None, ylabel=None):
    arr = df[column_name].tolist()
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
    freq_sorted = {k:v for k,v in sorted(freq.items(), key=lambda itm:itm[0])}
    x = list(freq_sorted.keys())
    y = list(freq_sorted.values())

    if logscale_x:
        values = []
        for i in range(len(x)-1):
            values += [(x[i+1]+x[i])/2] * y[i]
        mode = stats.mode(values,keepdims=False)
        mean = np.mean(values)
        median = np.median(values)
        skew = stats.skew(values)
        kurtosis = stats.kurtosis(values)
        print(f"Histogram (log xscale): mean={mean} mode={mode.mode} median={median} skew={skew} kurtosis={kurtosis}")
        xx = bin_width / 2
        plt.axvline(x = mean - xx, color = 'blue',   label="Mean")
        plt.axvline(x = median - xx, color = 'orange', label="Median")
        plt.axvline(x = mode.mode - xx, color = 'orange',    label="Mode")
        plt.legend()

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
        xticks.append(f"{x10[-1]}-")
        plt.xticks(x, xticks, rotation=rotation, ha="right")
    else:
        xticks = list(map(lambda t:round(t), x))
        plt.xticks(x, xticks, rotation=rotation, ha="right")
    if filename:
        plt.savefig(f"{file_location}/{filename}", bbox_inches="tight")
    plt.show()    

def est_type(etype):
    d = {"A": "Store",
        "B": "Bakery",
        "C": "Food Manufacturer",
        "D": "Food Warehouse",
        "E": "Beverage Plant",
        "F": "Feed Mill/Non-Medicated",
        "G": "Processing Plant",
        "H": "Wholesale Manufacturer",
        "I": "Refrigerated Warehouse",
        "J": "Multiple Operations",
        "K": "Vehicle",
        "L": "Produce Refrigerated Warehouse",
        "M": "Salvage Dealer",
        "N": "Wholesale Produce Packer",
        "O": "Produce Grower/Packer/Broker, Storage",
        "P": "C.A. Room",
        "Q": "Feed Mill/Medicated",
        "R": "Pet Food Manufacturer",
        "S": "Feed Warehouse and/or Distributor",
        "T": "Disposal Plant",
        "U": "Disposal Plant/Transportation Service",
        "V": "Slaughterhouse",
        "W": "Farm Winery-Exempt 20-C, for OCR Use",
        "Z": "Farm Product Use Only",
        "other": "other"}
    return d[etype]

def check_county(df):
    show_frequency(df, "County", rotation=45, top_n=20,
                   title="Number of Retail Stores per County",
                   ylabel="Number of Retail Stores")
    df2 = pd.read_csv("data/ny_county_population_2020.csv")
    c_sorted = df2["County"].tolist()
    p_sorted = df2["Population"].tolist()
    population = {c:p for c,p in zip(c_sorted,p_sorted)}
    stores = defaultdict(int)
    for c in df["County"].tolist():
        stores[c] += 1   
    s_sorted = {population[c]:stores[c] for c in c_sorted}
    x = list(s_sorted.keys())
    y = list(s_sorted.values())

    regressor = HuberRegressor(epsilon=1) # or LinearRegression()
    regressor.fit(np.array(x).reshape(-1,1),
                  np.array(y).reshape(-1))
    y_pred = regressor.predict(np.array(x).reshape(-1,1))
    coef = round(regressor.coef_[0],5)
    intc = round(regressor.intercept_,5)
    plt.plot(x,y_pred,"k",label=f"Predicted line: Y={coef}X+{intc}")

    # scatter plot for all the counties
    plt.scatter(x,y,label="County")
    plt.title("County Population vs Number of Retail Stores")
    plt.xlabel("County Population")
    plt.ylabel("Number of Retail Stores")
    plt.xticks(np.linspace(0,2500000,6),
               ["0","500,000","1,000,000","1,500,000","2,000,000","2,500,000"])
    plt.grid()
    plt.legend()
    plt.show()

def check_establishment_type(df):
    show_frequency(df,"Establishment Type", rotation=45, top_n=20,
                   title="Number of Retail Stores vs Establishment Type",
                   ylabel="Number of stores")
    ets = df["Establishment Type"].tolist()
    freq2 = defaultdict(int)
    for i in range(len(ets)):
        for j in range(len(ets[i])):
            freq2[ets[i][j]] += 1
    top_n = 10
    freq2_sorted = {k:v for k,v in sorted(freq2.items(), key=lambda itm:itm[1], reverse=True)}
    x = list(freq2_sorted.keys())[:top_n] + ["other"]
    y_last = sum(list(freq2_sorted.values())[top_n:])
    y = list(freq2_sorted.values())[:top_n] + [y_last]
    plt.bar(x,y)
    plt.title("Frequency of Retail Store Business type")
    plt.xlabel("Retail Store Business type")
    plt.ylabel("Frequency")
    plt.xticks(x,map(lambda e:est_type(e),x), rotation=45, ha="right")
    plt.grid()
    plt.show()

    freq3 = defaultdict(int)
    for i in range(len(ets)):
        freq3[len(ets[i])] += 1
    top_n = 10
    freq3_sorted = {k:v for k,v in sorted(freq3.items(), key=lambda itm:itm[1], reverse=True)}
    x = list(freq3_sorted.keys())
    y = list(freq3_sorted.values())
    plt.bar(x,y)
    plt.title("Frequency of Establishment Type length")
    plt.xlabel("Establishment Type length")
    plt.ylabel("Frequency")
    plt.show()

def check_entity_name(df):
    show_frequency(df,"Entity Name", rotation=45, top_n=20, drop_other=True,
                   title="20 most frequently appearing Retail Stores by Entity name",
                   xlabel="Entity Name",
                   ylabel="Frequency")

def check_dba_name(df):
    show_frequency(df,"DBA Name", rotation=45, top_n=20, drop_other=True,
                   title="20 most frequently appearing Retail Stores by DBA name",
                   xlabel="DBA Name",
                   ylabel="Frequency")

def check_city(df):
    show_frequency(df,"City", rotation=45, top_n=30,
                   title="Cities vs Retail Stores",
                   xlabel="City",
                   ylabel="Number of Retail Stores")

def check_zip_code(df):
    show_frequency(df, "Zip Code", top_n=20, drop_other=False, rotation=45)

def check_square_footage(df):
    show_frequency(df, "Square Footage", rotation=45, top_n=20, drop_other=False,
                   title="Frequently vs Square Footage of retail stores",
                   xlabel="Square Footage (Sqf)",
                   ylabel="Frequency")
    show_histogram(df, "Square Footage", logscale=False, logscale_x=False, rotation=45,
                   title="Frequently vs Square Footage of retail stores",
                   xlabel="Square Footage (Sqf)",
                   ylabel="Frequency")
    show_histogram(df, "Square Footage", logscale=False, logscale_x=True, rotation=45,
                   title="Frequently vs Square Footage (Log scale) of retail stores",
                   xlabel="Log scale of Square Footage (Sqf)",
                   ylabel="Frequency")

def main():    
    data_file = "data/Retail_Food_Stores.csv"
    df = pd.read_csv(data_file)

    # There are two missing values in the CITY column
    # https://opengovny.com/food-store/750317
    # https://opengovny.com/food-store/752951
    for i in range(len(df)):
        col_city = 10 # City is the 10th column
        row = df.iloc[i,:]
        if row["License Number"] == 750317:
            df.iat[i,col_city] = "SHELTER ISLAND"
        if row["License Number"] == 752951:
            df.iat[i,col_city] = "NEW YORK"
    
    # (1) Detect the variables and data types
    print("+++++++++++ Data type of each column +++++++++++")
    print(df.dtypes)
    print("+++++++++++ Number of Unique values in each column +++++++++++")
    print(df.nunique())
    
    # (2) Determine the shape of the data
    print("+++++++++++ Shape of the dataset +++++++++++")
    print("Number of rows:", len(df))
    print("Number of columns:", len(df.columns))
    
    # (3) Check for missing data and anormalies
    print("+++++++++++ Number of missing values in each column +++++++++++")
    print(df.isna().sum())

    # (4) Check each column
    check_county(df)
    check_establishment_type(df)
    check_entity_name(df)
    check_dba_name(df)
    check_city(df)
    check_zip_code(df)
    check_square_footage(df)

main()
