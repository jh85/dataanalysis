import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal

# Data sites:
# https://www.ncei.noaa.gov/pub/data/uscrn/products/subhourly01/2006/CRNS0101-05-2023-RI_Kingston_1_NW.txt
# https://www.data.jma.go.jp/obd/stats/etrn/view/daily_s1.php?prec_no=44&block_no=47662&year=1875&month=06&day=01&view=p1

def load_data_US(name):
    path = "/path/to/"
    df = pd.DataFrame()
    for y in range(2006,2024):
        filename = None
        if name == "RI":
            filename = f"{path}CRNS0101-05-{y}-RI_Kingston_1_NW.txt"
        elif name == "NH":
            filename = f"{path}CRNS0101-05-{y}-NH_Durham_2_SSW.txt"
        elif name == "FL":
            filename = f"{path}CRNS0101-05-{y}-FL_Titusville_7_E.txt"
        elif name == "ME":
            filename = f"{path}CRNS0101-05-{y}-ME_Limestone_4_NNW.txt"
        elif name == "VA":
            filename = f"{path}CRNS0101-05-{y}-VA_Cape_Charles_5_ENE.txt"
        elif name == "WY":
            filename = f"{path}CRNS0101-05-{y}-WY_Moose_1_NNE.txt"
        elif name == "AZ":
            filename = f"{path}CRNS0101-05-{y}-AZ_Tucson_11_W.txt"
        elif name == "HI":
            filename = f"{path}CRNS0101-05-{y}-HI_Mauna_Loa_5_NNE.txt"
        df_tmp = pd.read_csv(filename, dtype=object, delim_whitespace=True, header=None)
        df = pd.concat([df, df_tmp])
    columns = ['WBANNO', 'UTC_DATE', 'UTC_TIME', 'LST_DATE', 'LST_TIME', 'CRX_VN',
               'LONGITUDE', 'LATITUDE', 'AIR_TEMPERATURE', 'PRECIPITATION',
               'SOLAR_RADIATION', 'SR_FLAG', 'SURFACE_TEMPERATURE', 'ST_TYPE',
               'ST_FLAG', 'RELATIVE_HUMIDITY', 'RH_FLAG', 'SOIL_MOISTURE_5',
               'SOIL_TEMPERATURE_5', 'WETNESS', 'WET_FLAG', 'WIND_1_5', 'WIND_FLAG']
    df.columns = columns

    for c in ["AIR_TEMPERATURE","PRECIPITATION","SOLAR_RADIATION"]:
        df[c] = df[c].replace({"-9999.0":np.nan})
        df[c] = df[c].astype("float")
        df[c] = df[c].interpolate()
        df[c] = df[c].interpolate(method="linear", limit_direction="backward")

    date = []
    for i in range(len(df)):
        row = df.iloc[i,:]
        d = row["LST_DATE"]
        t = row["LST_TIME"]
        dt = d[:4] + "-" + d[4:6] + "-" + d[6:8] + " " + t[:2] + ":" + t[2:]
        date.append(dt)
    df["DATE"] = date
    temp = df["AIR_TEMPERATURE"].tolist()
    dT = 1/12 # 5 minutes, or 1/12 hour
    return temp,date,dT

def clean_data(arr1):
    arr2 = []
    for v in arr1:
        v = v.replace(" )", "")
        v = v.replace("#","")
        v = v.replace("///","")
        v = v.replace("×","") # Unicode
        v = v.replace("--","")
        try:
            v2 = float(v)
        except:
            v2 = np.nan
        arr2.append(v2)
    return arr2

def load_data_nonUS(name):
    path = "/path/to/"
    filename = None
    dT = None
    if name == "o":
        filename = f"{path}o.csv"
        columns = ["temperature", "precipitation", "humidity"]
        cname = "temperature"
        dT = 1/6 # every 10 minutes
    elif name == "s":
        filename = f"{path}south_pole.csv"
        columns = ["temperature", "precipitation", "humidity"]
        cname = "temperature"
        dT = 1 # hourly
    elif name == "t":
        filename = f"{path}t.csv"
        columns = ["temp_avg", "temp_max", "temp_min", "precipitation", "humidity"]
        cname = "temp_avg"
        dT = 24 # daily
    df = pd.read_csv(filename, dtype=object, low_memory=False)
    for c in columns:
        arr1 = df[c].astype("str").tolist()
        df = df.drop(columns=[c])
        df[c] = clean_data(arr1)
        df[c] = df[c].interpolate()
        df[c] = df[c].interpolate(method="linear", limit_direction="backward")
    temp = df[cname].tolist()
    date = df["datetime"].tolist()
    return temp,date,dT

def main():
    # alter describe() format
    pd.set_option("display.float_format", lambda x: "%.1f" % x)

    if False:
        for c in ["t","o","s"]:
            temp,date,dT = load_data_nonUS(c)
            df = pd.DataFrame(temp, columns=["temperature"])
            print(df["temperature"].describe())
        return

    if False:
        df = pd.DataFrame()
        for c in ["RI","NH","VA","FL","WY","HI","AZ","ME"]:
            temp,date,dT = load_data_US(c)
            df[c] = temp[:1886844]
        return

    if False:
        temp,x,dT = load_data_nonUS("t")
        # convert to datetime64
        x = np.asarray(x, dtype='datetime64[s]')
        plt.figure(figsize=(8,6))
        plt.scatter(x[0::1],temp[0::1],s=1)
        plt.title("Daily temperature of T")
        plt.xlabel("Time")
        plt.ylabel("Temperature")
        plt.grid()
        plt.show()
        return

        
    temp,date,dT = load_data_US("AZ")
    
    tmp_a = signal.detrend(temp)
    N = len(tmp_a)
    domg = 2*np.pi/(N*dT)        # minimal frequency: Delta omega
    yfft = np.fft.fft(tmp_a) / N

    # convert yfft to power spectral density (PSD)
    pw = np.abs(yfft)**2
    psd = pw / domg

    # one side PSD
    psd_oneside = psd[:N//2+1] * 2
    if N % 2 == 0:
        psd_oneside[N//2] /= 2
    psd_oneside[0] /= 2

    xfft = np.linspace(0.0, domg*(N/2), N//2+1)    
    # or equivalent
    # xfft2 = 2 * np.pi * np.fft.fftfreq(N, d=dT)
    # xfft = xfft2[0:N//2+1]

    # Convert freq to period
    xfft_T = 2*np.pi / xfft
    
    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(111)

    # PSD
    ax.plot(xfft_T, psd_oneside, label="PSD")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.invert_xaxis() # invert x-axis
    ax.set_title("Power Spectral Density of Temperature in Kingston RI") # ,size="x-large")
    ax.set_xlabel("Time (Hour)")
    ax.set_ylabel("Power Spectral Density")
    ax.legend(loc=2)
    ax.grid(True, which="both")
    plt.show()

    # Parseval's theorem: data's variance == Integral of PSD
    # variance
    var = tmp_a.var()
    # Integral of PSD
    var2 = (psd_oneside*domg).sum()
    print("variance =", var)
    print("Integral PSD=", var2)

main()
