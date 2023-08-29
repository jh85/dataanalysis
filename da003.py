import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import seaborn as sns
from matplotlib.colors import ListedColormap
from datetime import datetime
from scipy.stats import norm

# data: COVID-19 Tests, Cases, and Deaths by Town (Connecticut)
# URL: https://data.ct.gov/Health-and-Human-Services/COVID-19-Tests-Cases-and-Deaths-By-Town-ARCHIVE/28fr-iqnx
# file: COVID-19_Tests__Cases__and_Deaths__By_Town__-_ARCHIVE.csv

# global data
town_population = dict()

def town2population(town):
    # URL: https://portal.ct.gov/DPH/Health-Information-Systems--Reporting/Population/Annual-Town-and-County-Population-for-Connecticut
    # Title: Annual Town and County Population for Connecticut (2020)
    # Using the data, this function returns a dictionary of {"townname" : population}. The "townname" is lower-case.
    global town_population
    if len(town_population) == 0:
        data_file = "data/ct_population_2020.csv"
        df = pd.read_csv(data_file)
        for i in range(len(df)):
            town_population[df.iloc[i,0].lower()] = df.iloc[i,1]
    return town_population[town.lower()]

def sturges_rule(size):
    return 1 + int(np.log2(size))

def town_number2name(df, town_number):
    return df[df["Town number"]==town_number]["Town"].to_numpy()[0]

def normalize(arr):
    arr = np.array(arr)
    mu = np.mean(arr)
    sigma = np.std(arr)
    return (arr - mu) / sigma

def town_wise(df, town_numbers, column_name, conv2rate=False):
    for town_number in town_numbers:
        dates_string = df[df["Town number"] == town_number]["Last update date"].tolist()
        values       = df[df["Town number"] == town_number][column_name].tolist()
        if conv2rate:
            population = town2population(town_number2name(df, town_number))
            values = list(map(lambda v:v/population, values))
        dates = map(lambda s:datetime.strptime(s, '%m/%d/%Y').date(), dates_string)
        data = zip(dates, values)
        data_sorted   = sorted(data, key=lambda a:a[0])
        dates_sorted  = list(map(lambda v:v[0], data_sorted))
        values_sorted = list(map(lambda v:v[1], data_sorted))
        plt.plot(dates_sorted, values_sorted, "-", label=town_number2name(df,town_number))
    rate = " rate" if conv2rate else ""    
    plt.title(f"Daily trend of {column_name}{rate}")
    plt.ylabel(f"{column_name}{rate}")
    plt.xticks(rotation=45, ha="right")
    plt.grid()
    if len(town_numbers) < 5:
        plt.legend()
    plt.show()
    return

def town_wise_diff(df, town_numbers, column_name, conv2rate=False):
    for town_number in town_numbers:
        dates_string = df[df["Town number"] == town_number]["Last update date"].tolist()
        values       = df[df["Town number"] == town_number][column_name].tolist()
        if conv2rate:
            population = town2population(town_number2name(df, town_number))
            values = list(map(lambda v:v/population, values))
        dates = list(map(lambda s:datetime.strptime(s, '%m/%d/%Y').date(), dates_string))

        values_diff = []
        for i in range(len(dates)-1):
            val = max(0, (values[i+1] - values[i]) / (dates[i+1] - dates[i]).days)
            values_diff.append(val)        
        dates = dates[:-1]
        obsv = zip(dates, values_diff)
        obsv_sorted   = sorted(obsv, key=lambda a:a[0])
        dates_sorted  = list(map(lambda v:v[0], obsv_sorted))
        values_sorted = list(map(lambda v:v[1], obsv_sorted))
        plt.plot(dates_sorted, values_sorted, "-", label=town_number2name(df,town_number))

    rate = " rate" if conv2rate else ""
    plt.title(f"Daily trend of {column_name} increase{rate}")
    plt.ylabel(f"Daily{rate} increase")
    plt.xticks(rotation=45, ha="right")
    plt.grid()
    plt.show()
    return

def town_wise_diff_freq(df, town_numbers, column_name, conv2rate=False):
    for town_number in town_numbers:
        dates_string = df[df["Town number"] == town_number]["Last update date"].tolist()
        values       = df[df["Town number"] == town_number][column_name].tolist()
        if conv2rate:
            population = town2population(town_number2name(df, town_number))
            values = list(map(lambda v:v/population, values))

        dates = list(map(lambda s:datetime.strptime(s, '%m/%d/%Y').date(), dates_string))
        obsv = zip(dates, values)
        obsv_sorted   = sorted(obsv, key=lambda a:a[0])
        dates_sorted  = list(map(lambda v:v[0], obsv_sorted))
        values_sorted = list(map(lambda v:v[1], obsv_sorted))

        values_diff = []
        for i in range(len(dates)-1):
            val = max(0, (values_sorted[i+1] - values_sorted[i]) / (dates_sorted[i+1] - dates_sorted[i]).days)
            values_diff.append(val)        
        plt.hist(values_diff, bins=sturges_rule(len(values_diff)), log=False)

    rate = " rate" if conv2rate else ""
    plt.title(f"Frequency of daily{rate} change in {column_name}")
    plt.xlabel(f"Daily{rate} change in {column_name}")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()

    median = np.percentile(values_diff, 50)
    q1     = np.percentile(values_diff, 25)
    q3     = np.percentile(values_diff, 75)
    mu     = round(np.mean(values_diff),2)
    sigma  = round(np.std(values_diff),2)
        
    print(f"Daily changes of {column_name} in " + town_number2name(df,town_number))
    print(f"  mean: {mu}")
    print(f"  standard deviation: {sigma}")
    print(f"  q1: {q1}")
    print(f"  median: {median}")
    print(f"  q3: {q3}")
        
    return

def show_nullity_matrix(df):
    isnulls = df.isnull()
    hsv_modified = plt.get_cmap("gray", 256)
    cmap = ListedColormap(hsv_modified(np.linspace(0.7, 0.9, 2)))
    sns.heatmap(isnulls, cmap=cmap, linewidths=0, xticklabels=1, yticklabels=10000, linecolor="black")
    for i in range(isnulls.shape[1]+1):
        plt.axvline(i, color="black", lw=0.1)
    plt.tight_layout()
    plt.ylabel("Row index")
    plt.title("Nullity matrix (Null values are white)")
    plt.xticks(rotation=45, ha="right")
    plt.show()
    return

def check_last_update_date(df):
    values = df["Last update date"].to_numpy()
    values = list(map(lambda s:datetime.strptime(s, '%m/%d/%Y').date(), values))
    bins = sturges_rule(len(values))
    plt.hist(values, bins=bins)
    plt.title("Frequency of Last update date")
    plt.xlabel("Last update date")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.grid()
    plt.show()
    return

def check_town_number(df):
    values = df["Town number"].to_numpy()
    bins = sturges_rule(len(values))
    binwidth = (max(values) - min(values)) // bins
    plt.hist(values, bins=range(min(values), max(values) + binwidth, binwidth))
    plt.title("Frequency of Town number")
    plt.xlabel("Town number")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()

    freq = dict()
    for i in range(np.min(values), np.max(values)+1):
        freq[i] = 0
    for v in values:
        freq[v] += 1
    freq_sorted = {k:v for k,v in sorted(freq.items(), key=lambda item:item[0])}
    x = freq_sorted.keys()
    y = freq_sorted.values()
    plt.plot(x,y)
    plt.xlabel("Town number")
    plt.ylabel("Number of observations")
    plt.title("Number of observations per town")
    plt.grid()
    plt.show()
    return

def check_total_cases(df):
    c = "Total cases"
    values = df[c].to_numpy()
    bins = sturges_rule(len(values))
    binwidth = (max(values) - min(values)) // bins
    plt.hist(values, bins=range(min(values), max(values) + binwidth, binwidth))
    plt.title("Frequency of Total cases")
    plt.xlabel(c)
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()

    # Total cases column is accumulated case counts. It's not meaningful to calculate the mean or the variance.
    # Town numbers
    # 15: Bridgeport (largest town)
    # 21: Canaan (lowest cases rate)
    # 145: Union (smallest town)
    town_wise(df, [15], c)
    town_wise_diff(df, [15], c)
    town_wise_diff_freq(df, [15], c)
    town_wise(df,[145], c)
    town_wise_diff(df, [145], c)
    town_wise_diff_freq(df, [145], c)
    town_wise(df, [21,145], "Total cases", conv2rate=False)
    town_wise(df, [21,145], "Total cases", conv2rate=True)

    return

def check(df, c):
    values = df[c].dropna().tolist()

    median = np.percentile(values, 50)
    q1     = np.percentile(values, 25)
    q3     = np.percentile(values, 75)
    mu     = round(np.mean(values),2)
    sigma  = round(np.std(values),2)

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [1,2.5]})

    ax[0].boxplot(values, 0, 'gD', vert=False, showfliers=False)
    ax[0].axvline(q1,     color="black", alpha=1, linewidth=0.7, label=f"q1: {q1}")
    ax[0].axvline(median, color="red",   alpha=1, linewidth=1,   label=f"median: {median}")
    ax[0].axvline(q3,     color="green", alpha=1, linewidth=0.7, label=f"q3: {q3}")
    ax[0].set_title(f"Frequency of {c}")
    ax[0].axis('off')
    legend = ax[0].legend(framealpha=0.0)

    bins = sturges_rule(len(values))
    binwidth = (max(values) - min(values)) // bins
    ax[1].hist(values, bins=np.arange(min(values), max(values) + binwidth, binwidth))
    ax[1].set_xlabel(c)
    ax[1].set_ylabel("Frequency")
    ax[1].grid()
    plt.show()
        
    print(f"Statistics of {c}")
    print(f"  mean: {mu}")
    print(f"  standard deviation: {sigma}")
    print(f"  q1: {q1}")
    print(f"  median: {median}")
    print(f"  q3: {q3}")
    return

def all_graphs(df):
    columns = ['Total cases','Confirmed cases', 'Probable cases', 'Total deaths', 'Confirmed deaths',
               'Probable deaths', 'People tested','Number of tests', 'Number of positives',
               'Number of negatives', 'Number of indeterminates']
    columns2 = ['Case rate', 'Rate tested per 100k']
    towns = list(np.arange(1,170))
    for c in columns:
        town_wise(df, towns, c, conv2rate=True)
        town_wise_diff(df, towns, c, conv2rate=True)
    for c in columns2:
        town_wise(df, towns, c, conv2rate=False)
        town_wise_diff(df, towns, c, conv2rate=False)

def main():
    data_file = "data/COVID-19_Tests__Cases__and_Deaths__By_Town__-_ARCHIVE.csv"
    df = pd.read_csv(data_file)

    # The column name "Total cases" has a space at the end. Removing it.
    df = df.rename(columns={"Total cases ": "Total cases"})

    # (1) Detect the variables and data types
    print("+++++++++++ Dataset statistics +++++++++++")
    print(f"Number of variables: {len(df.columns)}")
    print(f"Number of observations: {len(df)}")

    # Filling the empty values by 0 creates artificial deltas on 05/31/2020
    # Therefore, not doing it.
    if False:
        df = df.fillna({"Confirmed cases":0,
                        "Probable cases":0,
                        "Case rate":0,
                        "Confirmed deaths":0,
                        "Probable deaths":0,
                        "People tested":0,
                        "Rate tested per 100k":0,
                        "Number of tests":0,
                        "Number of positives":0,
                        "Number of negatives":0,
                        "Number of indeterminates":0})

    # (2) Check for missing data and anormalies
    print("+++++++++++ Number of missing values in each column +++++++++++")
    print(f"Missing cells:")
    print(df.isna().sum())

    # visualize the location of missing data
    show_nullity_matrix(df)

    # Verifying where the missing data are located
    # They are between 04/24/2020 and 05/31/2020
    dates = set()
    for i in range(len(df)):
        if pd.isnull(df.iloc[i,:]).any():
            dates.add(df.iloc[i,:]["Last update date"])
    dates = sorted(list(dates))
    print(f"Missing data are seen between {dates[0]} and {dates[-1]}")

    # (3) Check for data types
    print("+++++++++++ Data types of each column +++++++++++")
    print(df.dtypes)
    # "Last update date" is Date. Town is string. The rest are integers.

    # Check each column one by one
    print("+++++++++++ Checking Last Update Date +++++++++++")
    check_last_update_date(df)

    print("+++++++++++ Checking Town number +++++++++++")
    check_town_number(df)

    print("+++++++++++ Checking Total cases +++++++++++")
    check_total_cases(df)

    print("+++++++++++ Checking the rest of the columns +++++++++++")
    columns = ['Confirmed cases', 'Probable cases', 'Case rate', 'Total deaths', 'Confirmed deaths',
               'Probable deaths', 'People tested', 'Rate tested per 100k', 'Number of tests',
               'Number of positives','Number of negatives', 'Number of indeterminates']
    for c in columns:
        check(df, c)

    all_graphs(df)
    print("done")     

main()
