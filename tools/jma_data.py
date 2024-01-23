from bs4 import BeautifulSoup  

import numpy as np
import pandas as pd
import json
import requests
from datetime import datetime  as dt, timedelta, datetime, timezone
import time
from urllib.error import HTTPError
from urllib.error import URLError
import sys

def get_10min_data(prec_no, block_no, location, start_date, end_date):
    base_url = "https://www.data.jma.go.jp/obd/stats/etrn/view/10min_s1.php?prec_no=%s&block_no=%s&year=%s&month=%s&day=%s&view="
    skip_header = 2

    sday = dt.strptime(start_date, "%Y-%m-%d")
    eday = dt.strptime(end_date, "%Y-%m-%d")

    cday = sday
    weathers = []
    while True:
        if cday == eday:
            break
        print(f"fetching {cday}")
        
        rows = None
        try:
            url = base_url % (prec_no, block_no, cday.year, cday.month, cday.day)
            resp = requests.get(url)
            resp.encoding = resp.apparent_encoding
            soup = BeautifulSoup(resp.text, "html.parser")
            rows = soup.findAll("tr", class_="mtx")
        except:
            print("Failed: url = {url}")
            cday = cday + timedelta(1)
            continue

        # skip the headers
        rows = rows[skip_header:]
        for row in rows:
            items = row.findAll("td")
            hhmm = items[0].text
            date_time = None
            if(hhmm == "24:00"):
                d = cday + timedelta(1)
                date_time = dt.strptime("%s-%s-%s %s" % (d.year, d.month, d.day, "00:00"), "%Y-%m-%d %H:%M")
            else:
                date_time = dt.strptime("%s-%s-%s %s" % (cday.year, cday.month, cday.day, hhmm), "%Y-%m-%d %H:%M")

            local_pressure     = items[1].text
            sealevel_pressure  = items[2].text
            precipitation      = items[3].text
            temperature        = items[4].text
            humidity           = items[5].text
            avg_wind_speed     = items[6].text
            avg_wind_direction = items[7].text
            max_wind_speed     = items[8].text
            max_wind_direction = items[9].text
            d_weather = [date_time,
                         prec_no,
                         block_no,
                         temperature,
                         precipitation,
                         humidity,
                         local_pressure,
                         sealevel_pressure,
                         avg_wind_speed,
                         avg_wind_direction,
                         max_wind_speed,
                         max_wind_direction]
            weathers.append(d_weather)

        cday = cday + timedelta(1)
    rec_size = len(weathers[0])
    m = np.array(weathers).reshape(-1,rec_size)
    df = pd.DataFrame(m)
    df.columns = ["datetime", "prec_no", "block_no", "temperature", "precipitation", "humidity",
                  "local_pressure", "sealevel_pressure", "avg_wind_speed", "avg_wind_direction",
                  "max_wind_speed", "max_wind_direction"]
    filename = f"jmadata_{location}_10min_{start_date}_{end_date}.csv"
    df.to_csv(filename, index=False)

def get_hourly_data(prec_no, block_no, location, start_date, end_date):
    base_url = "https://www.data.jma.go.jp/obd/stats/etrn/view/hourly_s1.php?prec_no=%s&block_no=%s&year=%s&month=%s&day=%s&view=p1"
    skip_header = 2

    date_start = dt.strptime(start_date, "%Y-%m-%d")
    date_end = dt.strptime(end_date, "%Y-%m-%d")

    date = date_start
    weathers = []
    while True:
        if date == date_end:
            break
        print(f"fetching {date}")

        rows = None
        try:
            url = base_url % (prec_no, block_no, date.year, date.month, date.day)
            r = requests.get(url)
            r.encoding = r.apparent_encoding
            soup = BeautifulSoup(r.text, "html.parser")
            rows = soup.findAll("tr", class_="mtx")
        except:
            print("Failed: url = {url}")
            date = date + timedelta(1)
            continue

        # skip the headers
        rows = rows[skip_header:]
        for row in rows:
            items = row.findAll("td")
            hhmm = items[0].text
            hhmm = hhmm.zfill(2) + ":00"
            date_time = None
            if(hhmm == "24:00"):
                d = date + timedelta(1)
                date_time = dt.strptime("%s-%s-%s %s" % (d.year, d.month, d.day, "00:00"), "%Y-%m-%d %H:%M")
            else:
                date_time = dt.strptime("%s-%s-%s %s" % (date.year, date.month, date.day, hhmm), "%Y-%m-%d %H:%M")

            local_pressure       = items[1].text
            sealevel_pressure    = items[2].text
            precipitation        = items[3].text
            temperature          = items[4].text
            dewpoint_temperature = items[5].text
            vapor_pressure       = items[6].text
            humidity             = items[7].text
            wind_speed           = items[8].text
            wind_direction       = items[9].text
            insolation_duration  = items[10].text
            solar_irradiance     = items[11].text
            snowfall             = items[12].text
            snow_depth           = items[13].text
            weather              = items[14].text
            cloud_cover          = items[15].text
            visibility           = items[16].text
            d_weather = [date_time,
                         prec_no,
                         block_no,
                         temperature,
                         precipitation,
                         humidity,
                         local_pressure,
                         sealevel_pressure,
                         wind_speed,
                         wind_direction,
                         insolation_duration,
                         solar_irradiance,
                         snowfall,
                         snow_depth,
                         weather,
                         cloud_cover,
                         visibility]
            weathers.append(d_weather)

        date = date + timedelta(1)

    rec_size = len(weathers[0])
    m = np.array(weathers).reshape(-1,rec_size)
    df = pd.DataFrame(m)
    df.columns = ["datetime", "prec_no", "block_no", "temperature", "precipitation", "humidity",
                  "local_pressure", "sealevel_pressure", "wind_speed", "wind_direction",
                  "insolation_duration", "solar_irradiance", "snowfall", "snow_depth", "weather",
                  "cloud_cover", "visibility"]
    filename = f"jmadata_{location}_hourly_{start_date}_{end_date}.csv"
    df.to_csv(filename, index=False)

def get_10min_data_all():
    if False:
        # Mt. Fuji
        prec_no = "50"
        block_no = "47639"
        start_date = "2009-02-01"
        end_date = "2023-12-20"
        location = "mt_fuji"
        get_10min_data(prec_no, block_no, location, start_date, end_date)

    if False:
        # Maebashi
        prec_no = "42"
        block_no = "47624"
        start_date = "2008-07-01"
        end_date = "2023-12-20"
        location = "maebashi"
        get_10min_data(prec_no, block_no, location, start_date, end_date)

    if False:
        # Nigata
        prec_no = "54"
        block_no = "47604"
        start_date = "2008-07-01"
        end_date = "2023-12-20"
        location = "nigata"
        get_10min_data(prec_no, block_no, location, start_date, end_date)

    if False:
        # Oki Island Shimane
        prec_no = "68"
        block_no = "47740"
        start_date = "2008-07-01"
        end_date = "2023-12-20"
        location = "oki_island"
        get_10min_data(prec_no, block_no, location, start_date, end_date)

    if False:
        # Cape Muroto
        prec_no = "74"
        block_no = "47899"
        start_date = "2008-07-01"
        end_date = "2023-12-20"
        location = "muroto_misaki"
        get_10min_data(prec_no, block_no, location, start_date, end_date)

    if False:
        # Osaka
        prec_no = "62"
        block_no = "47772"
        start_date = "2008-07-01"
        end_date = "2023-12-19"
        location = "osaka"
        get_10min_data(prec_no, block_no, location, start_date, end_date)

    if False:
        # Tokyo
        prec_no = "44"
        block_no = "47662"
        start_date = "2008-07-01"
        end_date = "2023-12-19"
        location = "tokyo"
        get_10min_data(prec_no, block_no, location, start_date, end_date)
        
    if False:
        # Wakkanai
        prec_no = "11"
        block_no = "47401"
        start_date = "2008-07-01"
        end_date = "2023-12-19"
        location = "wakkanai"
        get_10min_data(prec_no, block_no, location, start_date, end_date)

    if False:
        # Showa station (Antarctica)
        prec_no = "99"
        block_no = "89532"
        start_date = "2016-02-01"
        end_date = "2023-12-19"
        location = "showa"
        get_10min_data(prec_no, block_no, location, start_date, end_date)

    if False:
        # Yonaguni Island
        prec_no = "91"
        block_no = "47912"
        start_date = "2008-07-01"
        end_date = "2023-12-19"
        location = "yonaguni_island"
        get_10min_data(prec_no, block_no, location, start_date, end_date)

    if False:
        # Minamitori Island
        prec_no = "44"
        block_no = "47991"
        start_date = "2010-06-01"
        end_date = "2023-12-19"
        location = "minamitori_island"
        get_10min_data(prec_no, block_no, location, start_date, end_date)

    if False:
        # Nara
        prec_no = "64"
        block_no = "47780"
        start_date = "2008-07-01"
        end_date = "2023-12-19"
        location = "nara"
        get_10min_data(prec_no, block_no, location, start_date, end_date)


def get_hourly_data_all():
    if False:
        # Mt. Fuji
        prec_no = "50"
        block_no = "47639"
        start_date = "1991-01-01"
        end_date = "2023-12-20"
        location = "mt_fuji"
        get_hourly_data(prec_no, block_no, location, start_date, end_date)

    if False:
        # Maebashi
        prec_no = "42"
        block_no = "47624"
        start_date = "1990-04-01"
        end_date = "2023-12-20"
        location = "maebashi"
        get_hourly_data(prec_no, block_no, location, start_date, end_date)

    if False:
        # Nigata
        prec_no = "54"
        block_no = "47604"
        start_date = "1990-04-01"
        end_date = "2023-12-20"
        location = "nigata"
        get_hourly_data(prec_no, block_no, location, start_date, end_date)

    if False:
        # Oki Island Shimane
        prec_no = "68"
        block_no = "47740"
        start_date = "1990-04-01"
        end_date = "2023-12-20"
        location = "oki_island"
        get_hourly_data(prec_no, block_no, location, start_date, end_date)

    if False:
        # Cape Muroto
        prec_no = "74"
        block_no = "47899"
        start_date = "1990-04-01"
        end_date = "2023-12-20"
        location = "muroto_misaki"
        get_hourly_data(prec_no, block_no, location, start_date, end_date)

    if False:
        # Osaka
        prec_no = "62"
        block_no = "47772"
        start_date = "1990-04-01"
        end_date = "2023-12-19"
        location = "osaka"
        get_hourly_data(prec_no, block_no, location, start_date, end_date)

    if False:
        # Tokyo
        prec_no = "44"
        block_no = "47662"
        start_date = "1990-01-01"
        end_date = "2023-12-19"
        location = "tokyo"
        get_hourly_data(prec_no, block_no, location, start_date, end_date)
        
    if False:
        # Wakkanai
        prec_no = "11"
        block_no = "47401"
        start_date = "1990-05-01"
        end_date = "2023-12-19"
        location = "wakkanai"
        get_hourly_data(prec_no, block_no, location, start_date, end_date)

    if False:
        # Showa station (Antarctica)
        prec_no = "99"
        block_no = "89532"
        start_date = "1989-04-01"
        end_date = "2023-12-19"
        location = "showa"
        get_hourly_data(prec_no, block_no, location, start_date, end_date)

    if False:
        # Yonaguni Island
        prec_no = "91"
        block_no = "47912"
        start_date = "1990-04-01"
        end_date = "2023-12-19"
        location = "yonaguni_island"
        get_hourly_data(prec_no, block_no, location, start_date, end_date)

    if False:
        # Minamitori Island
        prec_no = "44"
        block_no = "47991"
        start_date = "1990-01-02" # 1990-01-08 has bad data
        end_date = "2023-12-19"
        location = "minamitori_island"
        get_hourly_data(prec_no, block_no, location, start_date, end_date)

    if False:
        # Nara
        prec_no = "64"
        block_no = "47780"
        start_date = "1990-05-01"
        end_date = "2023-12-19"
        location = "nara"
        get_hourly_data(prec_no, block_no, location, start_date, end_date)

def main():
    get_10min_data_all()
    get_hourly_data_all()

main()
