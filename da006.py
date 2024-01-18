import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as signal
import scipy.stats as stats
import datetime
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from collections import Counter
from sklearn.mixture import GaussianMixture
import geopandas as gpd
from shapely.geometry import Point,Polygon
import os
from io import StringIO

# File locations
DATA_PATH = "/Users/user1/project1/"
DATA_PATH2 = f"{DATA_PATH}/small_data/"
TEMPERATURE_STATS = f"{DATA_PATH}/temperature_stats.csv"
PRESSURE_STATS = f"{DATA_PATH}/pressure_stats.csv"
WIND_SPEED_STATS = f"{DATA_PATH}/wind_speed_stats.csv"
AIRPORT_LIST = f"{DATA_PATH}/airports.csv"

# Available data 2000 ~ 2023
US_AIRPORTS = ["BIS","HON","GFK","LNK","DDC","GRI","MLI","PAKN","SAN","EVV","INL","DEN",
               "TOL","TYS","ABI","GNV","MSO","SHR","IPT","SGF","LAS","FAT","FSM","SMX",
               "MEM","ABQ","BRO","PNS","PGUM","ATL","BWI","BTR","PABT","BHM","BOI","BIL",
               "BUF","BTV","CPR","CHS","CRW","CLT","ORD","CVG","COS","CMH","DFW","HSE",
               "IAD","ELP","GGW","BDR","PHNL","CAR","IAH","IND","JAN","JFK","EYW","LIT",
               "BOS","LAX","PHTO","LEB","MKE","MSP","BNA","EWR","NEW","OKC","OMA","SEA",
               "PHL","PHX","ELY","PDX","PVD","SLC","TJSJ","FSD","STL","PABR","DCA","ICT",
               "ILG","WMC","ORH"]
# Available data 2005 ~ 2023
US_AIRPORTS2 = ["AIA","BZN","PAE","HFD","HUL","LAR","LXV","LOU","MHT","MOT","VUO","PHP",
                "PIR","MBS","PATA","PAWI","PAYA","VIH","BCE","CDC","RDM","P68","SNS",
                "SFF","MLP","GUY","BYI"]
# Available data 2006 ~ 2023
US_STATES = ["AK","AL","AR","AZ","CA","CO","FL","GA","HI","ID","IL","KS","LA","ME","MI",
             "MN","MO","MS","MT","NC","ND","NE","NH","NV","NY","OK","ON","OR","PA","RI",
             "SC","SD","TN","TX","VA","WA","WI","WV","WY"]
JMA1 = ["OS60","OS10","MB60","MB10","FJ60","FJ10","NI60","NI10","OI60","OI10","MU60",
        "MU10","TK60","TK10","WA60","WA10","SH60","SH10","MI60","MI10","NA60","NA10"]
JMA2 = ["TK24"]

def load_data_from_file_US(name):
    df = pd.DataFrame()
    # some of 2006 data are hourly. Not every 10min. Skipping 2006 data
    for y in range(2007,2024):
        filename = None
        if name == "AK":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-AK_Fairbanks_11_NE.txt"
        elif name == "AL":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-AL_Valley_Head_1_SSW.txt"
        elif name == "AR":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-AR_Batesville_8_WNW.txt"
        elif name == "AZ":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-AZ_Tucson_11_W.txt"
        elif name == "CA":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-CA_Stovepipe_Wells_1_SW.txt"
        elif name == "CO":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-CO_La_Junta_17_WSW.txt"
        elif name == "FL":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-FL_Titusville_7_E.txt"
        elif name == "GA":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-GA_Watkinsville_5_SSE.txt"
        elif name == "HI":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-HI_Mauna_Loa_5_NNE.txt"
        elif name == "ID":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-ID_Murphy_10_W.txt"
        elif name == "IL":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-IL_Shabbona_5_NNE.txt"
        elif name == "KS":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-KS_Oakley_19_SSW.txt"
        elif name == "LA":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-LA_Lafayette_13_SE.txt"
        elif name == "ME":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-ME_Limestone_4_NNW.txt"
        elif name == "MI":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-MI_Chatham_1_SE.txt"
        elif name == "MN":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-MN_Goodridge_12_NNW.txt"
        elif name == "MO":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-MO_Chillicothe_22_ENE.txt"
        elif name == "MS":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-MS_Newton_5_ENE.txt"
        elif name == "MT":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-MT_St._Mary_1_SSW.txt"
        elif name == "NC":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-NC_Asheville_13_S.txt"
        elif name == "ND":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-ND_Medora_7_E.txt"
        elif name == "NE":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-NE_Harrison_20_SSE.txt"
        elif name == "NH":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-NH_Durham_2_SSW.txt"
        elif name == "NV":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-NV_Mercury_3_SSW.txt"
        elif name == "NY":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-NY_Ithaca_13_E.txt"
        elif name == "OK":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-OK_Goodwell_2_E.txt"
        elif name == "ON":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-ON_Egbert_1_W.txt"
        elif name == "OR":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-OR_Corvallis_10_SSW.txt"
        elif name == "PA":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-PA_Avondale_2_N.txt"
        elif name == "RI":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-RI_Kingston_1_NW.txt"
        elif name == "SC":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-SC_McClellanville_7_NE.txt"
        elif name == "SD":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-SD_Sioux_Falls_14_NNE.txt"
        elif name == "TN":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-TN_Crossville_7_NW.txt"
        elif name == "TX":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-TX_Muleshoe_19_S.txt"
        elif name == "VA":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-VA_Cape_Charles_5_ENE.txt"
        elif name == "WA":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-WA_Quinault_4_NE.txt"
        elif name == "WI":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-WI_Necedah_5_WNW.txt"
        elif name == "WV":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-WV_Elkins_21_ENE.txt"
        elif name == "WY":
            filename = f"{DATA_PATH}/CRNS0101-05-{y}-WY_Moose_1_NNE.txt"
        else:
            print("wrong option")
            return
        df_tmp = pd.read_csv(filename, dtype=object, delim_whitespace=True, header=None)
        df = pd.concat([df, df_tmp])

    columns = ['WBANNO', 'UTC_DATE', 'UTC_TIME', 'LST_DATE', 'LST_TIME', 'CRX_VN',
               'LONGITUDE', 'LATITUDE', 'AIR_TEMPERATURE', 'PRECIPITATION',
               'SOLAR_RADIATION', 'SR_FLAG', 'SURFACE_TEMPERATURE', 'ST_TYPE',
               'ST_FLAG', 'RELATIVE_HUMIDITY', 'RH_FLAG', 'SOIL_MOISTURE_5',
               'SOIL_TEMPERATURE_5', 'WETNESS', 'WET_FLAG', 'WIND_1_5', 'WIND_FLAG']
    df.columns = columns
    # 22   WIND_1_5  [6 chars]  cols 127 -- 132 
    #      Average wind speed, in meters per second, at a height of 1.5 meters.

    return df

def load_data_US(name, cname):
    df = load_data_from_file_US(name)
    dat = pd.to_numeric(df[cname].to_numpy())
    date = []
    for i in range(len(df)):
        row = df.iloc[i,:]
        d = row["LST_DATE"]
        t = row["LST_TIME"]
        dt = d[:4] + "-" + d[4:6] + "-" + d[6:8] + " " + t[:2] + ":" + t[2:]
        unixsec = datetime.datetime.strptime(dt,"%Y-%m-%d %H:%M").replace(tzinfo=datetime.timezone.utc).timestamp()
        date.append(unixsec)
    dT = 1/12
    return dat,date,dT

def clean_data(arr1, nan_val=np.nan):
    arr2 = []
    for v in arr1:
        v = v.replace(" )", "")
        v = v.replace("#","")
        v = v.replace("///","")
        v = v.replace("×","")
        v = v.replace("--","")
        try:
            v2 = float(v)
        except:
            v2 = nan_val
        arr2.append(v2)
    return arr2

def clean_wind_dir(arr1):
    directions = {"NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90,
                  "ESE":112.5, "SE":135, "SSE":157.5, "S":180,
                  "SSW":202.5, "SW":225, "WSW":247.5, "W":270,
                  "WNW":292.5, "NW":315, "NNW":337.5, "N":  0,
                  "0":-1}
    arr2 = []
    for v in arr1:
        v2 = v.replace(" )", "")
        v2 = v2.replace(" ]","")
        v2 = v2.replace("nan","")
        v2 = v2.replace("#","")
        v2 = v2.replace("×","")
        v2 = v2.replace("--","")
        v2 = v2.replace("東","E")
        v2 = v2.replace("南","S")
        v2 = v2.replace("西","W")
        v2 = v2.replace("北","N")
        v2 = v2.replace("静穏","0")
        m = None
        for dir,num in directions.items():
            if v2 == dir:
                m = num
                break
        if m is None:
            m = -1
        arr2.append(m)
    return arr2

def load_data_nonUS(name, cname="temperature"):
    filename = None
    columns = None
    columns2 = None
    dT = None
    if name == "OS10":         # Osaka, Japan
        filename = f"{DATA_PATH}/osaka_10min.csv"
        sampling_rate = "10min"
    elif name == "TK10":        # Tokyo, Japan
        filename = f"{DATA_PATH}/tokyo_10min.csv"
        sampling_rate = "10min"
    elif name == "NA10":        # Nara, Japan
        filename = f"{DATA_PATH}/nara_10min.csv"
        sampling_rate = "10min"
    elif name == "WA10":        # Wakkanai, Japan
        filename = f"{DATA_PATH}/wakkanai_10min.csv"
        sampling_rate = "10min"
    elif name == "YO10":        # Yonaguni Island, Japan
        filename = f"{DATA_PATH}/yonaguni_island_10min.csv"
        sampling_rate = "10min"
    elif name == "MI10":        # Minamitori Island, Japan
        filename = f"{DATA_PATH}/minamitori_island_10min.csv"
        sampling_rate = "10min"
    elif name == "MB10":        # Maebashi, Japan
        filename = f"{DATA_PATH}/maebashi_10min.csv"
        sampling_rate = "10min"
    elif name == "FJ10":        # Mount Fuji, Japan
        filename = f"{DATA_PATH}/mt_fuji_10min.csv"
        sampling_rate = "10min"
    elif name == "NI10":        # Nigata, Japan
        filename = f"{DATA_PATH}/nigata_10min.csv"
        sampling_rate = "10min"
    elif name == "OI10":        # Oki Island, Japan
        filename = f"{DATA_PATH}/oki_island_10min.csv"
        sampling_rate = "10min"
    elif name == "MU10":        # Oki Island, Japan
        filename = f"{DATA_PATH}/muroto_misaki_10min.csv"
        sampling_rate = "10min"
    elif name == "SH10":        # Showa station, Antarctica
        filename = f"{DATA_PATH}/showa_10min.csv"
        sampling_rate = "10min"
    elif name == "FJ60":        # Mt Fuji, Japan
        filename = f"{DATA_PATH}/mt_fuji_hourly.csv"
        sampling_rate = "hourly"
    elif name == "MB60":        # Maebashi, Japan
        filename = f"{DATA_PATH}/maebashi_hourly.csv"
        sampling_rate = "hourly"
    elif name == "NI60":        # Nigata, Japan
        filename = f"{DATA_PATH}/nigata_hourly.csv"
        sampling_rate = "hourly"
    elif name == "OI60":        # Oki Island, Japan
        filename = f"{DATA_PATH}/oki_island_hourly.csv"
        sampling_rate = "hourly"
    elif name == "MU60":        # Cape Muroto, Japan
        filename = f"{DATA_PATH}/muroto_misaki_hourly.csv"
        sampling_rate = "hourly"
    elif name == "OS60":        # Osaka, Japan
        filename = f"{DATA_PATH}/osaka_hourly.csv"
        sampling_rate = "hourly"
    elif name == "TK60":        # Tokyo, Japan
        filename = f"{DATA_PATH}/tokyo_hourly.csv"
        sampling_rate = "hourly"
    elif name == "WA60":        # Wakkanai, Japan
        filename = f"{DATA_PATH}/wakkanai_hourly.csv"
        sampling_rate = "hourly"
    elif name == "SH60":        # Showa station Antarctica
        filename = f"{DATA_PATH}/showa_hourly.csv"
        sampling_rate = "hourly"
    elif name == "MI60":        # Minamitori Island, Japan
        filename = f"{DATA_PATH}/minamitori_island_hourly.csv"
        sampling_rate = "hourly"
    elif name == "NA60":        # Nara, Japan
        filename = f"{DATA_PATH}/nara_hourly.csv"
        sampling_rate = "hourly"
    elif name == "TK24":        # Tokyo, Japan 1875 ~ 2023
        filename = f"{DATA_PATH}/tokyo_daily.csv"
        sampling_rate = "daily"
    else:
        print("wrong option:", name)
        return

    df = pd.read_csv(filename, dtype=object, low_memory=False)

    columns1 = None
    columns2 = None
    dT = None
    if sampling_rate == "10min":
        columns1 = ["temperature", "precipitation", "humidity","local_p",
                    "sealevel_p","avg_wind_abs","max_wind_abs"]
        columns2 = ["avg_wind_dir","max_wind_dir"]
        dT = 1/6 # every 10 minutes
    elif sampling_rate == "hourly":
        columns1 = ["temperature", "precipitation", "humidity", "local_p", "sealevel_p",
                    "wind_abs", "insolation_du", "s_irradiance", "snowfall",
                    "snow_amount", "cloud_amount", "visibility"]
        columns2 = ["wind_dir"]
        dT = 1 # hourly
    elif sampling_rate == "daily":
        columns1 = ["temp_avg", "temp_max", "temp_min", "precipitation", "humidity"]
        columns2 = []
        dT = 24 # daily
    else:
        print(f"wrong sampling rate: {sampling_rate}")
        return

    for c in columns1:
        arr1 = df[c].astype("str").tolist()
        df = df.drop(columns=[c])
        if c == "precipitation":
            df[c] = clean_data(arr1,nan_val=0.0)
        else:
            df[c] = clean_data(arr1)

    for c in columns2:
        arr1 = df[c].astype("str").tolist()
        df = df.drop(columns=[c])
        df[c] = clean_wind_dir(arr1)

    dat = df[cname].tolist()
    date = df["datetime"].tolist()
    sec = None
    if name == "TK24":
        sec = list(map(lambda d:datetime.datetime.strptime(d,"%Y-%m-%d").replace(tzinfo=datetime.timezone.utc).timestamp(), date))
    else:
        sec = list(map(lambda d:datetime.datetime.strptime(d,"%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp(), date))
    return dat,sec,dT

def clean_mesonet_data(arr1):
    arr2 = []
    failed = set()
    for v in arr1:
        v = v.replace("M", "")
        try:
            v2 = float(v)
        except:
            failed.add(v)
            v2 = np.nan
        arr2.append(v2)
    print(f"failed = {failed}")
    return arr2

def load_mesonet_data(name, cname, only_location=False):
    # station, station_name, lat, lon, valid(UTC),
    # tmpf,dwpf,sknt,drct,gust_drct,gust_sknt,
    # vis1_coeff,vis1_nd,vis2_coeff,vis2_nd,vis3_coeff,vis3_nd,
    # ptype,precip,pres1,pres2,pres3
    filename = None
    if name == "BOS":
        filename = f"{DATA_PATH}/mesonet_logan_ma.csv"
    elif name == "COS":
        filename = f"{DATA_PATH}/mesonet_colorado_springs_co.csv"
    elif name == "JFK":
        filename = f"{DATA_PATH}/mesonet_jfk_ny.csv"
    elif name == "PHTO":
        filename = f"{DATA_PATH}/mesonet_lyman_field_hi.csv"
    elif name == "PGUM":
        filename = f"{DATA_PATH}/mesonet_agana_guam.csv"
    elif name == "SLC":
        filename = f"{DATA_PATH}/mesonet_salt_lake_city_ut.csv"
    elif name == "PABR":
        filename = f"{DATA_PATH}/mesonet_utqiagvik_ak.csv"
    elif name == "TJSJ":
        filename = f"{DATA_PATH}/mesonet_san_juan_puerto_rico.csv"
    elif name == "NEW":
        filename = f"{DATA_PATH}/mesonet_new_orleans_la.csv"
    elif name == "DFW":
        filename = f"{DATA_PATH}/mesonet_dallas_tx.csv"
    elif name == "ORH":
        filename = f"{DATA_PATH}/mesonet_worcester_ma.csv"
    elif name == "ATL":
        filename = f"{DATA_PATH}/mesonet_atlanta_ga.csv"
    elif name == "LAX":
        filename = f"{DATA_PATH}/mesonet_los_angeles_ca.csv"
    elif name == "CLT":
        filename = f"{DATA_PATH}/mesonet_charlotte_nc.csv"
    elif name == "ORD":
        filename = f"{DATA_PATH}/mesonet_chicago_il.csv"
    elif name == "BUF":
        filename = f"{DATA_PATH}/mesonet_buffalo_ny.csv"
    elif name == "IAH":
        filename = f"{DATA_PATH}/mesonet_houston_tx.csv"
    elif name == "ELP":
        filename = f"{DATA_PATH}/mesonet_el_paso_nm.csv"
    elif name == "CPR":
        filename = f"{DATA_PATH}/mesonet_casper_wy.csv"
    elif name == "WMC":
        filename = f"{DATA_PATH}/mesonet_winnemucca_nv.csv"
    elif name == "BOI":
        filename = f"{DATA_PATH}/mesonet_boise_id.csv"
    elif name == "SEA":
        filename = f"{DATA_PATH}/mesonet_seattle_tacoma_wa.csv"
    elif name == "GGW":
        filename = f"{DATA_PATH}/mesonet_glasgow_mt.csv"
    elif name == "MSP":
        filename = f"{DATA_PATH}/mesonet_minneapolis_mn.csv"
    elif name == "CVG":
        filename = f"{DATA_PATH}/mesonet_cincinnati_ky.csv"
    elif name == "ICT":
        filename = f"{DATA_PATH}/mesonet_wichita_ks.csv"
    elif name == "STL":
        filename = f"{DATA_PATH}/mesonet_st_louis_mo.csv"
    elif name == "BHM":
        filename = f"{DATA_PATH}/mesonet_birmingham_al.csv"
    elif name == "BNA":
        filename = f"{DATA_PATH}/mesonet_nashville_wv.csv"
    elif name == "CRW":
        filename = f"{DATA_PATH}/mesonet_charleston_wv.csv"
    elif name == "IAD":
        filename = f"{DATA_PATH}/mesonet_dulles_va.csv"
    elif name == "PHL":
        filename = f"{DATA_PATH}/mesonet_philadelphia_pa.csv"
    elif name == "BTR":
        filename = f"{DATA_PATH}/mesonet_baton_rouge_la.csv"
    elif name == "FSD":
        filename = f"{DATA_PATH}/mesonet_sioux_falls_sd.csv"
    elif name == "PDX":
        filename = f"{DATA_PATH}/mesonet_portland_or.csv"
    elif name == "OMA":
        filename = f"{DATA_PATH}/mesonet_omaha_ne.csv"
    elif name == "MKE":
        filename = f"{DATA_PATH}/mesonet_milwaukee_wi.csv"
    elif name == "EYW":
        filename = f"{DATA_PATH}/mesonet_key_west_fl.csv"
    elif name == "PHX":
        filename = f"{DATA_PATH}/mesonet_phoenix_az.csv"
    elif name == "CAR":
        filename = f"{DATA_PATH}/mesonet_caribou_me.csv"
    elif name == "BTV":
        filename = f"{DATA_PATH}/mesonet_burlington_vt.csv"
    elif name == "PABT":
        filename = f"{DATA_PATH}/mesonet_bettles_field_ak.csv"
    elif name == "LEB":
        filename = f"{DATA_PATH}/mesonet_lebanon_nh.csv"
    elif name == "EWR":
        filename = f"{DATA_PATH}/mesonet_newark_nj.csv"
    elif name == "PVD":
        filename = f"{DATA_PATH}/mesonet_providence_ri.csv"
    elif name == "BDR":
        filename = f"{DATA_PATH}/mesonet_bridgeport_ct.csv"
    elif name == "ILG":
        filename = f"{DATA_PATH}/mesonet_wilmington_de.csv"
    elif name == "JAN":
        filename = f"{DATA_PATH}/mesonet_jackson_ms.csv"
    elif name == "PHNL":
        filename = f"{DATA_PATH}/mesonet_honolulu_hi.csv"
    elif name == "OKC":
        filename = f"{DATA_PATH}/mesonet_oklahoma_city_ok.csv"
    elif name == "DCA":
        filename = f"{DATA_PATH}/mesonet_washington_dc.csv"
    elif name == "BIL":
        filename = f"{DATA_PATH}/mesonet_billings_mt.csv"
    elif name == "CHS":
        filename = f"{DATA_PATH}/mesonet_charleston_sc.csv"
    elif name == "IND":
        filename = f"{DATA_PATH}/mesonet_indianapolis_in.csv"
    elif name == "CMH":
        filename = f"{DATA_PATH}/mesonet_columbus_oh.csv"
    elif name == "LIT":
        filename = f"{DATA_PATH}/mesonet_little_rock_ar.csv"
    elif name == "BWI":
        filename = f"{DATA_PATH}/mesonet_baltimore_md.csv"
    elif name == "EYW":
        filename = f"{DATA_PATH}/mesonet_key_west_fl.csv"
    elif name == "BRO":
        filename = f"{DATA_PATH}/mesonet_brownsville_tx.csv"
    elif name == "ELY":
        filename = f"{DATA_PATH}/mesonet_yelland_field_nv.csv"
    elif name == "PNS":
        filename = f"{DATA_PATH}/mesonet_pensacola_fl.csv"
    elif name == "MEM":
        filename = f"{DATA_PATH}/mesonet_memphis_tn.csv"
    elif name == "ABQ":
        filename = f"{DATA_PATH}/mesonet_albuquerque_nm.csv"
    elif name == "TOL":
        filename = f"{DATA_PATH}/mesonet_toledo_oh.csv"
    elif name == "BIS":
        filename = f"{DATA_PATH}/mesonet_bismarck_nd.csv"
    elif name == "HON":
        filename = f"{DATA_PATH}/mesonet_huron_sd.csv"
    elif name == "GFK":
        filename = f"{DATA_PATH}/mesonet_grand_forks_nd.csv"
    elif name == "LNK":
        filename = f"{DATA_PATH}/mesonet_lincoln_ne.csv"
    elif name == "DDC":
        filename = f"{DATA_PATH}/mesonet_dodge_ks.csv"
    elif name == "GRI":
        filename = f"{DATA_PATH}/mesonet_grand_island_ne.csv"
    elif name == "MLI":
        filename = f"{DATA_PATH}/mesonet_moline_il.csv"
    elif name == "PAKN":
        filename = f"{DATA_PATH}/mesonet_king_salmon_ak.csv"
    elif name == "SAN":
        filename = f"{DATA_PATH}/mesonet_san_diego_ca.csv"
    elif name == "EVV":
        filename = f"{DATA_PATH}/mesonet_evansville_in.csv"
    elif name == "INL":
        filename = f"{DATA_PATH}/mesonet_international_falls_mn.csv"
    elif name == "DEN":
        filename = f"{DATA_PATH}/mesonet_denver_co.csv"
    elif name == "TOL":
        filename = f"{DATA_PATH}/mesonet_toledo_oh.csv"
    elif name == "TYS":
        filename = f"{DATA_PATH}/mesonet_knoxville_tn.csv"
    elif name == "ABI":
        filename = f"{DATA_PATH}/mesonet_abilene_tx.csv"
    elif name == "GNV":
        filename = f"{DATA_PATH}/mesonet_gainesville_fl.csv"
    elif name == "MSO":
        filename = f"{DATA_PATH}/mesonet_missoula_mt_csv"
    elif name == "SHR":
        filename = f"{DATA_PATH}/mesonet_sheridan_wy.csv"
    elif name == "IPT":
        filename = f"{DATA_PATH}/mesonet_williamsport_pa.csv"
    elif name == "SGF":
        filename = f"{DATA_PATH}/mesonet_springfield_mo.csv"
    elif name == "LAS":
        filename = f"{DATA_PATH}/mesonet_las_vegas_nv.csv"
    elif name == "FAT":
        filename = f"{DATA_PATH}/mesonet_fresno_ca.csv"
    elif name == "FSM":
        filename = f"{DATA_PATH}/mesonet_fort_smith_ar.csv"
    elif name == "SMX":
        filename = f"{DATA_PATH}/mesonet_santa_maria_ca.csv"
    elif name == "HSE":
        filename = f"{DATA_PATH}/mesonet_hatteras_nc.csv"

    elif name == "AIA":
        filename = f"{DATA_PATH2}/mesonet_alliance_ne.csv"
    elif name == "BZN":
        filename = f"{DATA_PATH2}/mesonet_bozeman_mt.csv"
    elif name == "PAE":
        filename = f"{DATA_PATH2}/mesonet_everett_wa.csv"
    elif name == "HFD":
        filename = f"{DATA_PATH2}/mesonet_hartford_ct.csv"
    elif name == "HUL":
        filename = f"{DATA_PATH2}/mesonet_houlton_me.csv"
    elif name == "LAR":
        filename = f"{DATA_PATH2}/mesonet_laramie_wy.csv"
    elif name == "LXV":
        filename = f"{DATA_PATH2}/mesonet_leadville_co.csv"
    elif name == "LOU":
        filename = f"{DATA_PATH2}/mesonet_louisville_ky.csv"
    elif name == "MHT":
        filename = f"{DATA_PATH2}/mesonet_manchester_nh.csv"
    elif name == "MOT":
        filename = f"{DATA_PATH2}/mesonet_minot_nd.csv"
    elif name == "VUO":
        filename = f"{DATA_PATH2}/mesonet_pearson_wa.csv"
    elif name == "PHP":
        filename = f"{DATA_PATH2}/mesonet_philip_sd.csv"
    elif name == "PIR":
        filename = f"{DATA_PATH2}/mesonet_pierre_sd.csv"
    elif name == "MBS":
        filename = f"{DATA_PATH2}/mesonet_saginaw_mi.csv"
    elif name == "PATA":
        filename = f"{DATA_PATH2}/mesonet_tanana_ak.csv"
    elif name == "PAWI":
        filename = f"{DATA_PATH2}/mesonet_wainwright_ak.csv"
    elif name == "PAYA":
        filename = f"{DATA_PATH2}/mesonet_yakutat_ak.csv"
    elif name == "VIH":
        filename = f"{DATA_PATH2}/mesonet_vichy_mo.csv"
    elif name == "BCE":
        filename = f"{DATA_PATH2}/mesonet_bryce_canyon_ut.csv"
    elif name == "CDC":
        filename = f"{DATA_PATH2}/mesonet_cedar_city_ut.csv"
    elif name == "RDM":
        filename = f"{DATA_PATH2}/mesonet_redmond_or.csv"
    elif name == "P68":
        filename = f"{DATA_PATH2}/mesonet_eureka_nv.csv"
    elif name == "SNS":
        filename = f"{DATA_PATH2}/mesonet_salinas_ca.csv"
    elif name == "SFF":
        filename = f"{DATA_PATH2}/mesonet_spokane_wa.csv"
    elif name == "MLP":
        filename = f"{DATA_PATH2}/mesonet_mullan_id.csv"
    elif name == "GUY":
        filename = f"{DATA_PATH2}/mesonet_guymon_ok.csv"
    elif name == "BYI":
        filename = f"{DATA_PATH2}/mesonet_burley_id.csv"
    else:
        print(f"wrong option {name}")
        return
    if only_location:
        f = open(filename)
        df = pd.read_csv(StringIO([next(f) for _ in range(2)][1]),dtype=object,header=None)
        f.close()
        return df.iloc[0].tolist()

    df = pd.read_csv(filename, dtype=object, low_memory=False)
    dat0 = df[cname].tolist()
    dat = []
    for i in range(len(dat0)):
        try:
            v = float(dat0[i])
        except:
            v = np.nan
        dat.append(v)
    
    date0 = df["valid(UTC)"].tolist()
    date = list(map(lambda d:datetime.datetime.strptime(d,"%Y-%m-%d %H:%M").replace(tzinfo=datetime.timezone.utc).timestamp(), date0))
    dT = 1/60
    return dat,date,dT

def calc_PSD(dat, date, dT, data_name, window_type="none"):
    N = len(dat)
    T = dT * N
    dat_a = signal.detrend(dat, type="constant")

    if window_type == "hanning":
        dat_a *= np.hanning(N)
    elif window_type == "hamming":
        dat_a *= np.hamming(N)
    else:
        pass
    
    yfft = np.fft.fft(dat_a) / N
    ps = yfft[:N//2+1]
    psd = (np.abs(ps)**2) * 2
    psd[0] = psd[0] / 2
    xfft_T = np.concatenate([np.array([np.nan]), T/np.arange(1,N//2+1)])

    # Identify peaks
    wsz = 20
    peaks = []
    for i in range(N//2+1):
        f = 0 if i-wsz < 0 else i-wsz
        t = N//2 if i+wsz > N//2 else i+wsz
        mid = np.median(psd[f:t+1])
        prominence = psd[i]/mid
        if prominence > 10:
            peaks.append((i,prominence))
    peaks_sorted = sorted(peaks, key=lambda itm: itm[1], reverse=True)

    # Remove duplicate peaks (distance is within 10%)
    peaks_T = []
    for (loc,prom) in peaks_sorted:
        is_dup = False
        for pT in peaks_T:
            if np.abs(xfft_T[loc] - pT) < 0.1*xfft_T[loc]:
                is_dup = True
                break
        if is_dup == False:
            peaks_T.append(xfft_T[loc])

    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(111)

    # Pick up only 10 peaks for graph
    psz = 10
    colors = ["red","green","orange","blue","pink",
              "tan","cyan","navy","lime","brown"]
    peak_num = min(psz,len(peaks_T))
    peaks_T += [peaks_T[-1]] * (psz-peak_num)    
    for i in range(psz):
        pT = peaks_T[i]
        value = round(pT/24,1)
        unit = "d"
        if value < 1:
            value = round(pT,1)
            unit = "h"
        ax.vlines(pT,0,1e9,color=colors[i],linestyles="dotted",label=f"{value} {unit}")

    ax.plot(xfft_T, psd, label="PSD")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_title(f"Power Spectral Density location={data_name} window={window_type}")
    ax.set_xlabel("Time (Hour)")
    ax.set_ylabel("Power Spectral Density")

    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    #ax.legend(loc=2)
    ax.grid(True, which="both")
    plt.show()

    # Check Parseval's theorem.
    # variance of detrended data
    variance = np.var(dat_a)
    # integral of PSD
    integral_psd = np.sum(psd)
    print("variance =", variance)
    print("Integral PSD=", integral_psd)
    return peaks_T[:psz]
    
# Pandas DataFrame ==> R DataFrame
def pyr_df(df):
    with localconverter(robjects.default_converter + pandas2ri.converter):
        df_r = robjects.conversion.py2rpy(df)
    return df_r

# R DataFrame ==> Pandas DataFrame
def rpy_df(df_imp):
    with localconverter(robjects.default_converter + pandas2ri.converter):
        df_imp_pd = robjects.conversion.rpy2py(df_imp)
    return df_imp_pd

def fill_missing_interpolate(dat, date, dT, bad_thr, method, od_func=None):
    interval = int(dT * 3600) # dT is the data interval in hour
    dat2 = [np.nan] if np.isnan(dat[0]) else [dat[0]]
    date2 = [date[0]]
    bad_values = []
    for i in range(1,len(date)):
        if date[i] - date[i-1] > interval+1:
            num = int(date[i] - date[i-1]) // interval - 1
            dat2  += [np.nan                   for _ in range(num)]
            date2 += [date[i-1]+(j+1)*interval for j in range(num)]
        if np.abs(dat[i]) > bad_thr:
            bad_values.append(dat[i])
            dat2.append(np.nan)
        else:
            dat2.append(dat[i])
        date2.append(date[i])
    nan_count = np.count_nonzero(np.isnan(dat2))
    print(f"Data: total:{len(dat2)} missing:{nan_count} ({round(nan_count/len(dat2)*100,3)}%) will be interpolated")

    dat2 = np.array(dat2)
    
    # run outlier detection if od_func is provided. 
    if od_func:
        nans = od_func(dat2)
        dat2[nans] = np.nan
        outlier_count = np.count_nonzero(np.isnan(dat2)) - nan_count
        print(f"Outlier detection: outliers:{outlier_count} rate: {round(outlier_count/len(dat2)*100,3)}")
    else:
        print("Outlier detection not performed")
    
    # Interpolate missing data
    if method == "linear":
        nans = np.isnan(dat2)
        nums = ~nans
        nans_loc = np.nonzero(nans)[0]
        nums_loc = np.nonzero(nums)[0]
        #dat2[Missing loc flags]= np.interp(location of missing, X, Y)  
        dat2[nans] = np.interp(nans_loc, nums_loc, dat2[nums])
        return dat2.tolist(),date2,dT
    elif method == "skip":
        dat2 = np.array(dat2)
        date2 = np.array(date2)
        nans = np.isnan(dat2)
        dat2 = dat2[~nans]
        date2 = date2[~nans]
        return dat2.tolist(),date2.tolist(),dT
    else:
        imputeTS = importr("imputeTS")
        R_interpolation = robjects.r["na_interpolation"]
        R_kalman = robjects.r["na.kalman"]
        R_ma = robjects.r["na_ma"]
        df = pd.DataFrame()
        df["dat"] = dat2
        df["date"] = date2
        df2 = None
        if method == "linear2":
            df2 = rpy_df(R_interpolation(pyr_df(df), option = 'linear'))
        elif method == "spline":
            df2 = rpy_df(R_interpolation(pyr_df(df), option = 'spline'))
        elif method == "mvavg":
            df2 = rpy_df(R_ma(pyr_df(df), k = 4))
        elif method == "kalman":
            df2 = rpy_df(R_kalman(pyr_df(df), model = "auto.arima"))
        else:
            print("wrong method: {method}")
            return
        dat2 = df2["dat"].tolist()
        date2 = df2["date"].tolist()
        return dat2,date2,dT

def plot_spectrogram(dat, date, dT, window, stride, c, window_type="none"):
    fft_sz = window//2+1 - 1
    fft_num = (len(dat) - window) // stride
    m = np.empty([fft_sz, fft_num])   # [Vertical, Horizontal]
    eps = 1e-14
    for i in range(fft_num):
        w = signal.detrend(dat[stride*i:stride*i+window], type="constant")
        if window_type == "hanning":
            w *= np.hanning(window)
        elif window_type == "hamming":
            w *= np.hamming(window)
        wfft = np.fft.fft(w)[1:fft_sz+1] / window
        psd = (np.abs(wfft) ** 2) * 2
        #psd[0] /= 2
        m[:,i] = np.log10(psd + eps)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.set_title(f"STFT Spectrogram location={c} window={window_type}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Length of Period (Hour)")

    freq = np.arange(fft_sz+1)
    freq[0] = 1
    periods = (dT*window) / freq
    times = np.arange(fft_num+1)
    dt_times = list(map(lambda ts:datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d"),
                        date[0::stride][:fft_num+1]))
    interval = len(times)//11
    ax.set_xticks(times[::interval], dt_times[::interval], rotation=45, ha="right")
    pm = ax.pcolormesh(times, periods, m, shading="flat", cmap="jet", vmin=m.min(), vmax=m.max())
    #ax.invert_yaxis()
    ax.set_yscale("log")
    fig.colorbar(pm)
    fig.tight_layout()
    plt.show()

def plot_data(dat, date, c, title, threshold=None):    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.set_title(f"{title} in {c}")
    ax.set_xlabel("Date")
    ax.set_ylabel(title)

    if threshold is not None:
        colors = np.array(["blue" for _ in range(len(dat))])
        dat = np.array(dat)
        colors[dat > threshold] = "red"
        ax.scatter(date,dat,s=2,color=colors)
    else:
        ax.scatter(date,dat,s=2)
    dt_date = list(map(lambda ts:datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d"),date))
    interval = len(date)//11
    ax.set_xticks(date[::interval], dt_date[::interval], rotation=45, ha="right")
    ax.grid()
    fig.tight_layout()
    plt.show()

def conv2celsius(dat):
    return list(map(lambda f:(f-32)*5/9, dat))

def conv2mps(dat):
    return list(map(lambda knot:knot*1852/3600, dat))

def conv2hPa(dat):
    return list(map(lambda iHg: 1013.25 * (iHg / 29.92), dat))

def plot_histogram(dat,c,graph_title):
    n_bins = int(np.log2(len(dat)))
    sns.histplot(dat,bins=n_bins)
    plt.title(f"Histogram {graph_title} in {c}")
    plt.xlabel(graph_title)
    plt.ylabel("Observation counts")
    plt.show()

def find_x(x, value_ranges, f):
    s = value_ranges[0]
    l = value_ranges[1]
    for _ in range(50):
        m = (l+s)/2
        if f(m) < x:
            s = m
        else:
            l = m
    return l

def outlier_detect_interpolate(dat, method):
    if method == "exp_zscore":
        mu = np.mean(dat)
        threshold = find_x(0.9995, [0,100], lambda x: 1-np.exp(-x/mu))
        print(f"mu:{mu} threshold:{threshold}")
        dat2 = np.array(dat)
        dat2[dat2 > threshold] = np.nan
        nans = np.isnan(dat2)
        print(f"mu:{mu} threshold:{threshold} dropped:{nans.sum()}")
        nums = ~nans
        nans_loc = np.nonzero(nans)[0]
        nums_loc = np.nonzero(nums)[0]
        #dat2[Missing loc flags]= np.interp(location of missing, X, Y)  
        dat2[nans] = np.interp(nans_loc, nums_loc, dat2[nums])
        return dat2.tolist()
    else:
        print(f"wrong option {method}")
        return dat

def od_weibull(dat):
    valid_data = dat[~np.isnan(dat)]
    params = stats.weibull_min.fit(valid_data, floc=0)
    valid_max = np.max(valid_data)
    valid_data_num = len(valid_data)
    coverage = 0.9995
    threshold = find_x(coverage,[0,valid_max*5],lambda x:stats.weibull_min.cdf(x,c=params[0],loc=params[1],scale=params[2])**valid_data_num)
    print(f"weibull: threshold = {threshold}")
    return dat > threshold

def calc_weibull_stats(dat):
    params = stats.weibull_min.fit(dat, floc=0)
    c = params[0]
    loc = params[1]
    scale = params[2]
    mean = np.mean(dat)
    std = np.std(dat)
    return mean,std,c,loc,scale

def calc_wind_speed():
    winds = []
    if True:
        for c in US_AIRPORTS:
            data_type = "sknt"
            graph_title = "Wind Speed (m/s)"
            bad_threshold = 1000
            print(f"++++++++++ Processing location:{c} data:{data_type} ++++++++++")
            dat0,date0,dT0 = load_mesonet_data(c,data_type)
            dat,date,dT = fill_missing_interpolate(dat0,date0,dT0,bad_threshold,"linear", od_weibull)
            dat = conv2mps(dat)
            plot_data(dat,date,c,graph_title)
            plot_histogram(dat,c,graph_title)
            calc_PSD(dat,date,dT,c,"none")
            #plot_spectrogram(dat,date,dT,int(24*365/dT), int(24*7/dT), c, "none")
            plot_spectrogram(dat,date,dT,int(24*30/dT), int(24*7/dT), c, "none")
            r1,r2,r3,r4,r5 = calc_weibull_stats(dat)
            N = len(dat)
            winds.append([c,N,dT,r1,r2,r3,r4,r5])
    
    if True:
        for c in JMA1:
            data_type = "avg_wind_abs"
            if "60" in c:
                data_type = "wind_abs"
            graph_title = "Wind Speed (m/s)"
            print(f"++++++++++ Processing location:{c} data:{data_type} ++++++++++")
            bad_threshold = 300
            dat0,date0,dT0 = load_data_nonUS(c, data_type)
            dat,date,dT = fill_missing_interpolate(dat0,date0,dT0,bad_threshold,"linear",od_weibull)
            plot_data(dat,date,c,graph_title)
            plot_histogram(dat,c,graph_title)
            calc_PSD(dat,date,dT,c,"none")
            #plot_spectrogram(dat,date,dT,int(24*365/dT), int(24*7/dT),c,"none")
            plot_spectrogram(dat,date,dT,int(24*30/dT), int(24*7/dT), c, "none")
            r1,r2,r3,r4,r5 = calc_weibull_stats(dat)
            N = len(dat)
            winds.append([c,N,dT,r1,r2,r3,r4,r5])

    df = pd.DataFrame(data=winds, columns=["location","N","dT","mean","std","shape","loc","scale"])
    df.to_csv(f"{DATA_PATH}/winds_weibull_stats.csv", index=False)
    return

def calc_temperature():    
    peak_df = pd.DataFrame()
    if True:
        for c in US_AIRPORTS:
            data_type = "tmpf"
            graph_title = "Temperature (C)"
            data_thr = 200
            print(f"++++++++++ Processing location:{c} data:{data_type} ++++++++++")
            dat0,date0,dT0 = load_mesonet_data(c,data_type)
            dat,date,dT = fill_missing_interpolate(dat0,date0,dT0,data_thr,"linear")
            dat = conv2celsius(dat)

            plot_histogram(dat,c,data_type)
            peaks10 = calc_PSD(dat,date,dT,c,"none")
            peak_df[c] = peaks10
            plot_data(dat,date,c,graph_title)
            plot_spectrogram(dat,date,dT,int(24*365/dT), int(24*7/dT), c, "none")

    if True:
        for c in JMA1:
            data_type = "temperature"
            print(f"++++++++++ Processing location:{c} data:{data_type} ++++++++++")
            bad_threshold = 200
            dat0,date0,dT0 = load_data_nonUS(c, data_type)
            dat,date,dT = fill_missing_interpolate(dat0,date0,dT0,bad_threshold,"linear")
            plot_histogram(dat,c,"Temperature")
            peaks10 = calc_PSD(dat,date,dT,c,"none")
            peak_df[c] = peaks10
            plot_data(dat,date,c,"Temperature")
            plot_spectrogram(dat,date,dT,int(24*365/dT), int(24*7/dT),c,"none")

    if True:
        for c in JMA2:
            data_type = "temp_avg"
            print(f"++++++++++ Processing location:{c} data:{data_type} ++++++++++")
            dat0,date0,dT0 = load_data_nonUS(c, data_type)
            bad_threshold = 200
            dat,date,dT = fill_missing_interpolate(dat0,date0,dT0,bad_threshold,"linear")
            peaks10 = calc_PSD(dat,date,dT,c)
            peak_df[c] = peaks10
            plot_data(dat,date,c,"Temperature")
            plot_spectrogram(dat,date,dT,int(24*365/dT), int(24/dT),c,"none")

    if True:
        for c in US_STATES:
            data_type = "AIR_TEMPERATURE"
            print(f"++++++++++ Processing location:{c} data:{data_type} ++++++++++")
            dat0,date0,dT0 = load_data_US(c, data_type)
            bad_threshold = 200
            dat,date,dT = fill_missing_interpolate(dat0,date0,dT0,bad_threshold,"linear")
            plot_histogram(dat,c,"Temperature")
            peaks10 = calc_PSD(dat,date,dT,c,"none")
            peak_df[c] = peaks10
            plot_data(dat,date,c,"Temperature")
            plot_spectrogram(dat,date,dT,int(24*365/dT), int(24*7/dT), c, "none")

    peak_df.to_csv(f"{DATA_PATH}/peak_temperature.csv", index=False)
    return

def calc_pressure():
    if True:
        for c in US_AIRPORTS:
            data_type = "pres1"
            graph_title = "Pressure (hPa)"
            data_thr = 3000
            print(f"++++++++++ Processing location:{c} data:{data_type} ++++++++++")
            dat0,date0,dT0 = load_mesonet_data(c,data_type)
            dat,date,dT = fill_missing_interpolate(dat0,date0,dT0,data_thr,"linear")
            dat = conv2hPa(dat)
            plot_histogram(dat,c,data_type)
            calc_PSD(dat,date,dT,c,"none")
            plot_data(dat,date,c,graph_title)
            plot_spectrogram(dat,date,dT,int(24*365/dT), int(24*7/dT), c, "none")

    if True:
        for c in JMA1:
            data_type = "local_p"
            print(f"++++++++++ Processing location:{c} data:{data_type} ++++++++++")
            graph_title = "Pressure (hPa)"
            bad_threshold = 3000
            dat0,date0,dT0 = load_data_nonUS(c, data_type)
            dat,date,dT = fill_missing_interpolate(dat0,date0,dT0,bad_threshold,"linear")
            plot_histogram(dat,c,graph_title)
            calc_PSD(dat,date,dT,c,"none")
            plot_data(dat,date,c,graph_title)
            plot_spectrogram(dat,date,dT,int(24*365/dT), int(24*7/dT),c,"none")
    return

def gmm_graph(dat,date,data_type,c,n_components):
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(dat)
    mn = gmm.means_
    cv = gmm.covariances_
    sc = gmm.bic(dat)
    num = int(np.max(dat)-np.min(dat))
    plt.hist(dat, bins=num, density=True, alpha=0.5, color="b")
    x_range = np.linspace(np.min(dat), np.max(dat), num).reshape(-1, 1)
    pdf = np.exp(gmm.score_samples(x_range))
    plt.plot(x_range, pdf, color="red", lw=2)
    plt.title(f"{data_type} n_components={n_components} score={sc}")
    plt.vlines(mn,0,0.05)
    plt.show()
    return mn,cv,sc

    labels = gmm.predict(dat)    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.set_title(f"temerature {c}")
    ax.set_xlabel("Date")
    ax.set_ylabel(data_type)
    ax.scatter(date, dat, c=labels, s=2, alpha=0.5)
    dt_date = list(map(lambda ts:datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d"),date))
    interval = len(date)//11
    ax.set_xticks(date[::interval], dt_date[::interval], rotation=45, ha="right")
    ax.grid()
    fig.tight_layout()
    plt.show()
    return mn,cv,sc

def calc_pressure_gmm():
    if True:
        for c in US_AIRPORTS:
            data_type = "pres1"
            graph_title = "Pressure (hPa)"
            data_thr = 3000
            print(f"++++++++++ Processing location:{c} data:{data_type} ++++++++++")
            dat0,date0,dT0 = load_mesonet_data(c,data_type)
            dat,date,dT = fill_missing_interpolate(dat0,date0,dT0,data_thr,"skip")
            dat = conv2hPa(dat)
            dat = np.array(dat).reshape(-1,1)
            bic_scores = []
            for k in range(1,7):
                _,_,scores = gmm_graph(dat,date,graph_title,c,k)
                bic_scores.append(scores)
            print(bic_scores)
            plt.title(f"{graph_title} BIC score {c}")
            plt.plot(np.arange(1,7), bic_scores)
            plt.grid()
            plt.xlabel("Number of Normal Distributions")
            plt.ylabel("BIC score")
            plt.show()

    if True:
        for c in JMA1:
            data_type = "local_p"
            print(f"++++++++++ Processing location:{c} data:{data_type} ++++++++++")
            graph_title = "Pressure (hPa)"
            bad_threshold = 3000
            dat0,date0,dT0 = load_data_nonUS(c, data_type)
            dat,date,dT = fill_missing_interpolate(dat0,date0,dT0,bad_threshold,"skip")
            dat = np.array(dat).reshape(-1,1)
            bic_scores = []
            for k in range(1,7):
                _,_,scores = gmm_graph(dat,date,graph_title,c,k)
                bic_scores.append(scores)
            print(bic_scores)
            plt.title(f"{graph_title} BIC score {c}")
            plt.plot(np.arange(1,7), bic_scores)
            plt.grid()
            plt.xlabel("Number of Normal Distributions")
            plt.ylabel("BIC score")
            plt.show()
    return

def calc_temperature_gmm():
    if True:
        for c in US_AIRPORTS:
            data_type = "tmpf"
            bad_threshold = 200
            print(f"++++++++++ Processing location:{c} data:{data_type} ++++++++++")
            dat0,date0,dT0 = load_mesonet_data(c,data_type)
            dat,date,dT = fill_missing_interpolate(dat0,date0,dT0,bad_threshold,"skip")
            dat = conv2celsius(dat)
            dat = np.array(dat).reshape(-1,1)
            bic_scores = []
            for k in range(1,7):
                _,_,scores = gmm_graph(dat,date,graph_title,c,k)
                bic_scores.append(scores)
            print(bic_scores)
            plt.title(f"Temperature BIC score {c}")
            plt.plot(np.arange(1,7), bic_scores)
            plt.grid()
            plt.xlabel("Number of Normal Distributions")
            plt.ylabel("BIC score")
            plt.show()

    if True:
        for c in JMA1:
            data_type = "temperature"
            print(f"++++++++++ Processing location:{c} data:{data_type} ++++++++++")
            bad_threshold = 200
            dat0,date0,dT0 = load_data_nonUS(c, data_type)
            dat,date,dT = fill_missing_interpolate(dat0,date0,dT0,bad_threshold,"skip")
            dat = np.array(dat).reshape(-1,1)
            bic_scores = []
            for k in range(1,7):
                _,_,scores = gmm_graph(dat,date,graph_title,c,k)
                bic_scores.append(scores)
            print(bic_scores)
            plt.title(f"Temperature BIC score {c}")
            plt.plot(np.arange(1,7), bic_scores)
            plt.grid()
            plt.xlabel("Number of Normal Distributions")
            plt.ylabel("BIC score")
            plt.show()

    if True:
        for c in JMA2:
            data_type = "temp_avg"
            print(f"++++++++++ Processing location:{c} data:{data_type} ++++++++++")
            dat0,date0,dT0 = load_data_nonUS(c, data_type)
            bad_threshold = 200
            dat,date,dT = fill_missing_interpolate(dat0,date0,dT0,bad_threshold,"skip")
            dat = np.array(dat).reshape(-1,1)
            bic_scores = []
            for k in range(1,7):
                _,_,scores = gmm_graph(dat,date,graph_title,c,k)
                bic_scores.append(scores)
            print(bic_scores)
            plt.title(f"Temperature BIC score {c}")
            plt.plot(np.arange(1,7), bic_scores)
            plt.grid()
            plt.xlabel("Number of Normal Distributions")
            plt.ylabel("BIC score")
            plt.show()

    if True:
        for c in US_STATES:
            data_type = "AIR_TEMPERATURE"
            print(f"++++++++++ Processing location:{c} data:{data_type} ++++++++++")
            dat0,date0,dT0 = load_data_US(c, data_type)
            bad_threshold = 200
            dat,date,dT = fill_missing_interpolate(dat0,date0,dT0,bad_threshold,"skip")
            dat = np.array(dat).reshape(-1,1)
            bic_scores = []
            for k in range(1,7):
                _,_,scores = gmm_graph(dat,date,graph_title,c,k)
                bic_scores.append(scores)
            print(bic_scores)
            plt.title(f"Temperature BIC score {c}")
            plt.plot(np.arange(1,7), bic_scores)
            plt.grid()
            plt.xlabel("Number of Normal Distributions")
            plt.ylabel("BIC score")
            plt.show()

def get_statistics_data(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        return pd.DataFrame(data=[],columns=["name","mean","std"])

def add_statistics_data(df,data):
    idx = len(df)
    df.loc[idx] = data
    return df

def calc_wind_speed_mean():
    filename = WIND_SPEED_STATS
    df = get_statistics_data(filename)
    for c in US_AIRPORTS + US_AIRPORTS2:
        if c in df.name.values:
            continue
        data_type = "sknt"
        graph_title = "Wind Speed (m/s)"
        bad_threshold = 300
        print(f"++++++++++ Processing location:{c} data:{data_type} ++++++++++")
        dat0,date0,dT0 = load_mesonet_data(c,data_type)
        dat,date,dT = fill_missing_interpolate(dat0,date0,dT0,bad_threshold,"skip")
        dat = conv2mps(dat)
        m = np.mean(dat)
        s = np.std(dat)
        print(f"c = {c} mean = {m} std = {s}")
        add_statistics_data(df, [c,m,s])
    df.to_csv(filename, index=None)
    return df

def calc_temperature_mean():
    filename = TEMPERATURE_STATS
    df = get_statistics_data(filename)
    for c in US_AIRPORTS + US_AIRPORTS2:
        if c in df.name.values:
            continue
        data_type = "tmpf"
        graph_title = "Temperature"
        bad_threshold = 200
        print(f"++++++++++ Processing location:{c} data:{data_type} ++++++++++")
        dat0,date0,dT0 = load_mesonet_data(c,data_type)
        dat,date,dT = fill_missing_interpolate(dat0,date0,dT0,bad_threshold,"skip")
        dat = conv2celsius(dat)
        m = np.mean(dat)
        s = np.std(dat)
        print(f"c = {c} mean = {m} std = {s}")
        add_statistics_data(df, [c,m,s])
    df.to_csv(filename, index=None)
    return df

def calc_pressure_mean():
    filename = PRESSURE_STATS
    df = get_statistics_data(filename)
    for c in US_AIRPORTS + US_AIRPORTS2:
        if c in df.name.values:
            continue
        data_type = "pres1"
        graph_title = "Pressure (hPa)"
        bad_threshold = 4000
        print(f"++++++++++ Processing location:{c} data:{data_type} ++++++++++")
        dat0,date0,dT0 = load_mesonet_data(c,data_type)
        dat,date,dT = fill_missing_interpolate(dat0,date0,dT0,bad_threshold,"skip")
        dat = conv2hPa(dat)
        m = np.mean(dat)
        s = np.std(dat)
        print(f"c = {c} mean = {m} std = {s}")
        add_statistics_data(df, [c,m,s])
    df.to_csv(filename, index=None)
    return df

def tocolor(dat, d):
    red = ["#ffe6e6","#ffcccc","#ffb3b3","#ff9999","#ff8080",
           "#ff6666","#ff4d4d","#ff3333","#ff1a1a","#ff0000"]
    lo = np.min(d)
    hi = np.max(d)
    colors = []
    for v in dat:
        idx = int((v-lo)/(hi-lo)*99/10)
        c = red[idx]
        colors.append(c)
    return colors

def disp_infographics(df_loc, df_data, data_type):
    shapefile = f"{DATA_PATH}/cb_2018_us_state_500k.shp"
    states = gpd.read_file(shapefile)

    airport_df = pd.merge(df_loc, df_data, on="name")

    mainland = states[states.STUSPS != "PR"]
    mainland = mainland[mainland.STUSPS != "AK"]
    mainland = mainland[mainland.STUSPS != "HI"]
    mainland = mainland[mainland.STUSPS != "MP"]
    mainland = mainland[mainland.STUSPS != "GU"]
    mainland = mainland[mainland.STUSPS != "AS"]
    mainland = mainland[mainland.STUSPS != "VI"]
    ak       = states[states.STUSPS == "AK"]
    hi       = states[states.STUSPS == "HI"]

    mainland_df = airport_df[airport_df["name"]!="PGUM"]
    mainland_df = mainland_df[mainland_df["name"]!="PHNL"]
    mainland_df = mainland_df[mainland_df["name"]!="PHTO"]
    mainland_df = mainland_df[mainland_df["name"]!="TJSJ"]
    mainland_df = mainland_df[mainland_df["name"]!="PAKN"]
    mainland_df = mainland_df[mainland_df["name"]!="PABT"]
    mainland_df = mainland_df[mainland_df["name"]!="PANL"]
    mainland_df = mainland_df[mainland_df["name"]!="PABR"]
    mainland_df = mainland_df[mainland_df["name"]!="PATA"]
    mainland_df = mainland_df[mainland_df["name"]!="PAWI"]
    mainland_df = mainland_df[mainland_df["name"]!="PAYA"]
    ak_df       = airport_df[(airport_df["name"]=="PAKN")| \
                             (airport_df["name"]=="PABT")| \
                             (airport_df["name"]=="PANL")| \
                             (airport_df["name"]=="PABR")| \
                             (airport_df["name"]=="PATA")| \
                             (airport_df["name"]=="PAWI")| \
                             (airport_df["name"]=="PAYA")]
    hi_df       = airport_df[(airport_df["name"]=="PHNL")|(airport_df["name"]=="PHTO")]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")

    bgcolor = "olive"
    
    akax = fig.add_axes([0.16, 0.23, 0.17, 0.16])
    akax.axis("off")
    akpolygon = Polygon([(-170,50),(-170,74),(-140,74),(-140,50)])
    alaska_gdf = states[states.STUSPS=="AK"]
    alaska_gdf.clip(akpolygon).plot(ax=akax,color=bgcolor,edgecolor="black")
    
    hiax = fig.add_axes([.34, 0.26, 0.1, 0.1])
    hiax.axis("off")
    hipolygon = Polygon([(-160,0),(-160,90),(-120,90),(-120,0)])
    hawaii_gdf = states[states.STUSPS=="HI"]
    hawaii_gdf.clip(hipolygon).plot(ax=hiax,color=bgcolor,edgecolor="black")

    d = airport_df["mean"].tolist()
    colors = tocolor(mainland_df["mean"].to_numpy(), d)
    mksize = 100
    geometry = [Point(lon,lat) for lon,lat in zip(mainland_df["longitude"],mainland_df["latitude"])]
    airports = gpd.GeoDataFrame(mainland_df, geometry=geometry)
    mainland.plot(ax=ax, color=bgcolor, edgecolor="black")
    airports.plot(ax=ax, c=colors, markersize=mksize)

    akcolors = tocolor(ak_df["mean"].to_numpy(), d)
    akgeometry = [Point(lon,lat) for lon,lat in zip(ak_df["longitude"],ak_df["latitude"])]
    akairports = gpd.GeoDataFrame(ak_df, geometry=akgeometry)
    akairports.plot(ax=akax, c=akcolors, markersize=mksize)

    hicolors = tocolor(hi_df["mean"].to_numpy(), d)
    higeometry = [Point(lon,lat) for lon,lat in zip(hi_df["longitude"],hi_df["latitude"])]
    hiairports = gpd.GeoDataFrame(hi_df, geometry=higeometry)
    hiairports.plot(ax=hiax, c=hicolors, markersize=mksize)

    # Create colorbar legend
    fig = ax.get_figure()
    # l:left, b:bottom, w:width, h:height; in normalized unit (0-1)
    cbax = fig.add_axes([0.79, 0.27, 0.03, 0.24])   
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(vmin=np.min(d), vmax=np.max(d)))
    sm._A = []
    fig.colorbar(sm, cax=cbax)
    tick_font_size = 10
    cbax.tick_params(labelsize=tick_font_size)

    ax.set_title(f"Average {data_type} of US Airports\n2001 ~ 2023")
    plt.show()

def get_airport_list():
    airports = []
    for c in US_AIRPORTS + US_AIRPORTS2:
        loc = load_mesonet_data(c,"none",only_location=True)
        # station,station_name,lat,lon,valid(UTC),tmpf,dwpf,...
        airports.append([c,loc[0],loc[1],loc[2],loc[3]])
    df = pd.DataFrame(data=airports,columns=["name","station","station_name","latitude","longitude"])
    df.to_csv(AIRPORT_LIST, index=None)
    return df

def main():
    # calc_wind_speed()
    # calc_temperature()
    # calc_pressure()
    #calc_pressure_gmm()
    #calc_temperature_gmm()
    df_airport = get_airport_list()

    #df_data = calc_wind_speed_mean()
    #disp_infographics(df_airport, df_data, "Wind Speed (m/s)")
    
    #df_data = calc_temperature_mean()
    #disp_infographics(df_airport, df_data, "Temperature (C)")

    df_data = calc_pressure_mean()
    disp_infographics(df_airport, df_data, "Atmospheric Pressure (hPa)")

main()
