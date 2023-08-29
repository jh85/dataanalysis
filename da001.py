import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math

# data: New York City Leading Causes of Death
# URL: https://data.cityofnewyork.us/Health/New-York-City-Leading-Causes-of-Death/jb7j-dtam
# file: New_York_City_Leading_Causes_of_Death.csv

def sort_freq(freq,key):
    # Sort by key(key==0) in ascending order, and by value(key==1) in descending order
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
    plt.xticks(rotation=rotation, ha="right")
    plt.grid()
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()

def show_histogram(df, column_name, logscale=False, logscale_x=False, remove_na=False,
                   rotation=0, filename=None, title=None, xlabel=None, ylabel=None):
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
        xticks.append(f"{x10[-1]}-")
        plt.xticks(x, xticks, rotation=rotation, ha="right")
    else:
        xticks = list(map(lambda t:round(t), x))
        plt.xticks(x, xticks, rotation=rotation, ha="right")
    plt.grid()
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()

def check_year(df):
    show_frequency(df, "Year", sortby_x=True, rotation=45,
                   title="Number of Observations per Year",
                   ylabel="Number of Observations")

def cause_graph(df, c, value):
    is_rate = "Rate" in value
    
    total = defaultdict(float)
    mdata = defaultdict(float)
    fdata = defaultdict(float)
    for i in range(len(df)):
        row = df.iloc[i,:]
        year = row["Year"]
        cause = row["Leading Cause"]
        sex = row["Sex"]
        cnt = row[value]
        if cause == c:
            if math.isnan(cnt) != True:
                if is_rate == False:
                    total[year] += cnt
                if sex == "M":
                    mdata[year] += cnt
                else:
                    fdata[year] += cnt
    total_s = sort_freq(total,0)
    mdata_s = sort_freq(mdata,0)
    fdata_s = sort_freq(fdata,0)
    # Total doesn't make sense for Death Rate and Age Adjusted Death Rate
    if is_rate == False:
        plt.plot(total_s.keys(), total_s.values(), color="green", linestyle="solid", marker="o", label="Total")
    plt.plot(mdata_s.keys(), mdata_s.values(), color="blue", linestyle="solid", marker="o", label="Male")
    plt.plot(fdata_s.keys(), fdata_s.values(), color="orange", linestyle="solid", marker="o", label="Female")
    plt.title(f"Annual {value} of {c}")
    plt.xlabel("Year")
    plt.ylabel(f"Annual {value}")
    plt.legend()
    plt.grid()
    plt.xticks(range(2007,2020), rotation=45, ha="right")
    plt.xlim(2006.5,2019.5)
    plt.ylim(0)
    plt.show()

def check_leading_cause(df):
    # Plot histogram for "Leading Cause" with Y-axis showing the frequency count.
    show_frequency(df, "Leading Cause", rotation=45,
                   title="Number of observations per Leading Cause")

    # Plot another histogram for "Leading Cause" with Y-axis showing the total number of Deaths
    freq = defaultdict(int)
    for i in range(len(df)):
        row = df.iloc[i,:]
        lc = row["Leading Cause"]
        num = row["Deaths"]
        sex = row["Sex"]
        freq[lc] += num
    freq_s = sort_freq(freq,1)
    x = list(freq_s.keys())
    y = list(freq_s.values())
    plt.bar(x,y)
    plt.xticks(rotation=45, ha="right")
    plt.title("Number of deceased people per Leading Cause")
    plt.xlabel("Leading Cause")
    plt.ylabel("Number of deceased people")
    plt.yscale("log")
    plt.grid()
    plt.show()

    # Plot annual trend for each leading cause.
    columns = ["Cancer","Heart Disease","Flu/Pneumonia","Stroke",
               "CLRD","Hypertension","Substance","Accidents","Suicide",
               "Homicide","Liver Disease"]
    for c in columns:
        cause_graph(df,c,"Age Adjusted Death Rate") # "Age Adjusted Death Rate" vs. Year

def check_sex(df):
    show_frequency(df, "Sex", title="Number of gender data",
                   xlabel="Gender", ylabel="Number of data")

def racename(k):
    repl_race = {1:'Black Non-Hispanic',
                 2:'Not Stated/Unknown',
                 3:'Hispanic', 
                 4:'Asian and Pacific Islander', 
                 5:'White Non-Hispanic', 
                 6:'Other Race/ Ethnicity'}
    return repl_race[k]

def check_race(df):
    #
    # Plot frequency graph for Race/Ethnicity
    #
    arr = df["Race Ethnicity"].tolist()
    freq = defaultdict(int)
    for i in range(len(arr)):
        freq[arr[i]] += 1
    freq_sorted = sort_freq(freq,1)
    x = list(freq_sorted.keys())
    y = list(freq_sorted.values())
    plt.bar(x,y)
    orig  = [1,2,3,4,5,6]
    names = list(map(racename, orig))
    plt.xticks(orig, names, rotation=45, ha="right")
    plt.xlabel("Race / Ethnicity")
    plt.ylabel("Frequency")
    plt.title("Frequency of Race/Ethnicity")
    plt.grid()
    plt.show()

    #
    # Plot race-wise Age Adjusted Death Rate vs Year graph
    #
    freq2m = defaultdict(int) # unknown Male
    freq2f = defaultdict(int) #         Female
    freq4m = defaultdict(int) # asian Male
    freq4f = defaultdict(int) #       Female
    freq5m = defaultdict(int) # white Male
    freq5f = defaultdict(int) #       Female
    freq6m = defaultdict(int) # other Male
    freq6f = defaultdict(int) #       Female
    for i in range(len(df)):
        row = df.iloc[i,:]
        year = row["Year"]
        lc = row["Leading Cause"]
        num = row["Age Adjusted Death Rate"]
        sex = row["Sex"]
        race = row["Race Ethnicity"]
        if lc == "Suicide":
            if race == 2 and sex == "M": freq2m[year] = num
            if race == 2 and sex == "F": freq2f[year] = num
            if race == 4 and sex == "M": freq4m[year] = num
            if race == 4 and sex == "F": freq4f[year] = num
            if race == 5 and sex == "M": freq5m[year] = num
            if race == 5 and sex == "F": freq5f[year] = num
            if race == 6 and sex == "M": freq6m[year] = num
            if race == 6 and sex == "F": freq6f[year] = num
    freq2m = sort_freq(freq2m,0)
    freq2f = sort_freq(freq2f,0)
    freq4m = sort_freq(freq4m,0)
    freq4f = sort_freq(freq4f,0)
    freq5m = sort_freq(freq5m,0)
    freq5f = sort_freq(freq5f,0)
    freq6m = sort_freq(freq6m,0)
    freq6f = sort_freq(freq6f,0)
    # plt.plot(freq2m.keys(), freq2m.values(), linestyle="solid", marker="o", label=f"{racename(2)} M")
    # plt.plot(freq2f.keys(), freq2f.values(), linestyle="solid", marker="o", label=f"{racename(2)} F")
    plt.plot(freq4m.keys(), freq4m.values(), linestyle="solid", marker="o", label=f"{racename(4)} M")
    plt.plot(freq4f.keys(), freq4f.values(), linestyle="solid", marker="o", label=f"{racename(4)} F")
    plt.plot(freq5m.keys(), freq5m.values(), linestyle="solid", marker="o", label=f"{racename(5)} M")
    # There is no White F data in Suicide column
    # plt.plot(freq5f.keys(), freq5f.values(), linestyle="solid", marker="o", label=f"{racename(5)} F")
    # plt.plot(freq6m.keys(), freq6m.values(), linestyle="solid", marker="o", label=f"{racename(6)} M")
    # plt.plot(freq6f.keys(), freq6f.values(), linestyle="solid", marker="o", label=f"{racename(6)} F")
    plt.legend()
    plt.grid()
    plt.title("Annual Age Adjusted Death Rate of Suicides per Race/Gender")
    plt.xlabel("Year")
    plt.ylabel("Annual Age Adjusted Death Rate of Suicides")
    plt.xticks(range(2007,2020), rotation=45, ha="right")
    plt.xlim(2006.5,2019.5)
    plt.ylim(-1,16)
    plt.show()

    #
    # Race-wise Deaths vs Year Graph
    #
    freq2m = defaultdict(int) # unknown
    freq2f = defaultdict(int)
    freq4m = defaultdict(int) # asian
    freq4f = defaultdict(int)
    freq5m = defaultdict(int) # white
    freq5f = defaultdict(int)
    freq6m = defaultdict(int) # other
    freq6f = defaultdict(int)
    for i in range(len(df)):
        row = df.iloc[i,:]
        year = row["Year"]
        lc = row["Leading Cause"]
        num = row["Deaths"] 
        sex = row["Sex"]
        race = row["Race Ethnicity"]
        if lc == "Suicide":
            if race == 2 and sex == "M": freq2m[year] = num
            if race == 2 and sex == "F": freq2f[year] = num
            if race == 4 and sex == "M": freq4m[year] = num
            if race == 4 and sex == "F": freq4f[year] = num
            if race == 5 and sex == "M": freq5m[year] = num
            if race == 5 and sex == "F": freq5f[year] = num
            if race == 6 and sex == "M": freq6m[year] = num
            if race == 6 and sex == "F": freq6f[year] = num
    freq2m = sort_freq(freq2m,0)
    freq2f = sort_freq(freq2f,0)
    freq4m = sort_freq(freq4m,0)
    freq4f = sort_freq(freq4f,0)
    freq5m = sort_freq(freq5m,0)
    freq5f = sort_freq(freq5f,0)
    freq6m = sort_freq(freq6m,0)
    freq6f = sort_freq(freq6f,0)
    # plt.plot(freq2m.keys(), freq2m.values(), linestyle="solid", marker="o", label=f"{racename(2)} M")
    # plt.plot(freq2f.keys(), freq2f.values(), linestyle="solid", marker="o", label=f"{racename(2)} F")
    plt.plot(freq4m.keys(), freq4m.values(), linestyle="solid", marker="o", label=f"{racename(4)} M")
    plt.plot(freq4f.keys(), freq4f.values(), linestyle="solid", marker="o", label=f"{racename(4)} F")
    plt.plot(freq5m.keys(), freq5m.values(), linestyle="solid", marker="o", label=f"{racename(5)} M")
    # There is no White F data in Suicide column
    # plt.plot(freq5f.keys(), freq5f.values(), linestyle="solid", marker="o", label=f"{racename(5)} F")
    # plt.plot(freq6m.keys(), freq6m.values(), linestyle="solid", marker="o", label=f"{racename(6)} M")
    # plt.plot(freq6f.keys(), freq6f.values(), linestyle="solid", marker="o", label=f"{racename(6)} F")
    plt.legend()
    plt.grid()
    plt.title("Annual Number of Suicides per Race/Gender")
    plt.xlabel("Year")
    plt.ylabel("Annual Number of Suicides")
    plt.xticks(range(2007,2020), rotation=45, ha="right")
    plt.xlim(2006.5,2019.5)
    plt.ylim(0,225)
    plt.show()

    # calculate the correlation 
    # US overall data was taken from
    # https://www.cdc.gov/suicide/suicide-data-statistics.html
    frequs = {2007:11.3, 2008:11.6, 2009:11.8, 2010:12.1, 2011:12.3,
              2012:12.6, 2013:12.6, 2014:13.0, 2015:13.3, 2016:13.5,
              2017:14.0, 2018:14.2, 2019:13.9}
    freq4m = defaultdict(int) # asian
    freq4f = defaultdict(int)
    freq5m = defaultdict(int) # white
    for i in range(len(df)):
        row = df.iloc[i,:]
        year = row["Year"]
        lc = row["Leading Cause"]
        num = row["Death Rate"] 
        sex = row["Sex"]
        race = row["Race Ethnicity"]
        if lc == "Suicide":
            if race == 4 and sex == "M": freq4m[year] = num
            if race == 4 and sex == "F": freq4f[year] = num
            if race == 5 and sex == "M": freq5m[year] = num
    freq4m = sort_freq(freq4m,0)
    freq4f = sort_freq(freq4f,0)
    freq5m = sort_freq(freq5m,0)

    df_4m = pd.DataFrame(list(freq4m.values()), columns=["Asian/Islander M"])
    df_4f = pd.DataFrame(list(freq4f.values()), columns=["Asian/Islander F"])
    df_5m = pd.DataFrame(list(freq5m.values()), columns=["White Non-Hispanic M"])
    df_us = pd.DataFrame(list(frequs.values()), columns=["US overall"])
    df_all = pd.concat([df_4m,df_4f,df_5m,df_us], axis=1)
    print(df_all.corr())

    plt.plot(freq4m.keys(), freq4m.values(), linestyle="solid", marker="o", label=f"Asian/Islander M")
    plt.plot(freq4f.keys(), freq4f.values(), linestyle="solid", marker="o", label=f"Asian/Islander F")
    plt.plot(freq5m.keys(), freq5m.values(), linestyle="solid", marker="o", label=f"White Non-Hispanic M")
    plt.plot(frequs.keys(), frequs.values(), linestyle="solid", marker="o", label=f"US overall")
    plt.legend()
    plt.grid()
    plt.title("Annual Suicide Death Rate per Race/Gender")
    plt.xlabel("Year")
    plt.ylabel("Annual Suicide Death Rate")
    plt.xticks(range(2007,2020), rotation=45, ha="right")
    plt.xlim(2006.5,2019.5)
    plt.ylim(0,21)
    plt.show()

def main():
    data_file = "data/New_York_City_Leading_Causes_of_Death.csv"
    df = pd.read_csv(data_file)

    # Replace dot values with 0
    df["Sex"] = df["Sex"].replace(["Male","Female"],["M","F"])
    df["Deaths"] = df["Deaths"].replace(".", "0")
    df["Death Rate"] = df["Death Rate"].replace(".", "0")
    df["Age Adjusted Death Rate"] = df["Age Adjusted Death Rate"].replace(".", "0")

    # Fix typos and minor inconsistencies in Leading Cause
    df["Leading Cause"] = df["Leading Cause"].replace(["Accidents Except Drug Posioning (V01-X39, X43, X45-X59, Y85-Y86)",
                                                       "Intentional Self-Harm (Suicide: X60-X84, Y87.0)",
                                                       "Chronic Liver Disease and Cirrhosis (K70, K73)",
                                                       "Assault (Homicide: Y87.1, X85-Y09)"],
                                                      ["Accidents Except Drug Poisoning (V01-X39, X43, X45-X59, Y85-Y86)",
                                                       "Intentional Self-Harm (Suicide: U03, X60-X84, Y87.0)",
                                                       "Chronic Liver Disease and Cirrhosis (K70, K73-K74)",
                                                       "Assault (Homicide: U01-U02, Y87.1, X85-Y09)"])
    # Fix inconsistency in the Race Ethnicity column
    df["Race Ethnicity"] = df["Race Ethnicity"].replace(["Non-Hispanic White", "Non-Hispanic Black"],
                                                        ["White Non-Hispanic", "Black Non-Hispanic"])
    # Convert Race/Ethnicity to integer values for convenience
    df["Race Ethnicity"] = df["Race Ethnicity"].replace(["Black Non-Hispanic",
                                                         "Not Stated/Unknown",
                                                         "Hispanic",
                                                         "Asian and Pacific Islander",
                                                         "White Non-Hispanic",
                                                         "Other Race/ Ethnicity"],
                                                        [1,2,3,4,5,6])
    
    # Shorten the long names for convenience
    df["Leading Cause"] = df["Leading Cause"].replace(
        ["Malignant Neoplasms (Cancer: C00-C97)",
         "All Other Causes",
         "Diseases of Heart (I00-I09, I11, I13, I20-I51)",
         "Influenza (Flu) and Pneumonia (J09-J18)",
         "Diabetes Mellitus (E10-E14)",
         "Cerebrovascular Disease (Stroke: I60-I69)",
         "Chronic Lower Respiratory Diseases (J40-J47)",
         "Essential Hypertension and Renal Diseases (I10, I12)",
         "Mental and Behavioral Disorders due to Accidental Poisoning and Other Psychoactive Substance Use (F11-F16, F18-F19, X40-X42, X44)",
         "Accidents Except Drug Poisoning (V01-X39, X43, X45-X59, Y85-Y86)",
         "Alzheimer's Disease (G30)",
         "Human Immunodeficiency Virus Disease (HIV: B20-B24)",
         "Intentional Self-Harm (Suicide: U03, X60-X84, Y87.0)",
         "Certain Conditions originating in the Perinatal Period (P00-P96)",
         "Chronic Liver Disease and Cirrhosis (K70, K73-K74)",
         "Nephritis, Nephrotic Syndrome and Nephrisis (N00-N07, N17-N19, N25-N27)",
         "Assault (Homicide: U01-U02, Y87.1, X85-Y09)",
         "Septicemia (A40-A41)",
         "Congenital Malformations, Deformations, and Chromosomal Abnormalities (Q00-Q99)",
         "Mental and Behavioral Disorders due to Use of Alcohol (F10)",
         "Viral Hepatitis (B15-B19)",
         "Aortic Aneurysm and Dissection (I71)",
         "Insitu or Benign / Uncertain Neoplasms (D00-D48)",
         "Parkinson's Disease (G20)",
         "Atherosclerosis (I70)",
         "Peptic Ulcer (K25-K28)",
         "Cholelithiasis and Disorders of Gallbladder (K80-K82)",
         "Anemias (D50-D64)",
         "Tuberculosis (A16-A19)",
         "Complications of Medical and Surgical Care (Y40-Y84, Y88)",
         "Meningitis (G00, G03)",
         "Pregnancy, Childbirth and the Puerperium (O00-O09)"],
        ["Cancer","Other","Heart Disease","Flu/Pneumonia","Diabetes","Stroke",
         "CLRD","Hypertension","Substance","Accidents","Alzheimer","HIV","Suicide",
         "Perinatal Period","Liver Disease","Nephritis","Homicide","Septicemia",
         "Malformations","Alcohol","Hepatitis","Aneurysm","Neoplasms",
         "Parkinson's","Atherosclerosis","Peptic Ulcer",
         "Cholelithiasis","Anemias","Tuberculosis", 
         "Complications","Meningitis","Pregnancy"])
                                                          
    # Convert data types of quantitative values to numeric for convenience
    df = df.astype({"Deaths":"int",
                    "Death Rate":"float",
                    "Age Adjusted Death Rate":"float"})

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

    check_year(df)
    check_leading_cause(df)
    check_sex(df)
    check_race(df)
    show_histogram(df, "Deaths",logscale=True)
    show_histogram(df, "Death Rate", remove_na=True, logscale=True)
    show_histogram(df, "Age Adjusted Death Rate", remove_na=True, logscale=True)

main()
