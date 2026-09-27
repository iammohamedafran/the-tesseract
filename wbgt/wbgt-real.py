
import pandas as pd
import numpy as np

from pywbgt import wbgt
from metpy.units import units


# --------------------------------------------------
# 1. Read CSV
# --------------------------------------------------

df = pd.read_csv("final-final-final.csv")


# --------------------------------------------------
# 2. Fixed dates
# --------------------------------------------------

dates = np.array([
    "2026-05-30 12:00",
    "2026-05-31 12:00",
    "2026-06-01 12:00",
    "2026-06-02 12:00",
    "2026-06-03 12:00",
    "2026-06-04 12:00",
    "2026-06-05 12:00"
], dtype="datetime64")



# --------------------------------------------------
# 3. Calculate WBGT for every district
# --------------------------------------------------

all_wbgt = []


for index, row in df.iterrows():

    # Location
    latitude = np.repeat(row["latitude"], 7)
    longitude = np.repeat(row["longitude"], 7)

    # Weather data for 7 days
    temp_air = np.array([
        row["temp_D1"],
        row["temp_D2"],
        row["temp_D3"],
        row["temp_D4"],
        row["temp_D5"],
        row["temp_D6"],
        row["temp_D7"]
    ]) * units("degC")

    temp_dew = np.array([
        row["dew_D1"],
        row["dew_D2"],
        row["dew_D3"],
        row["dew_D4"],
        row["dew_D5"],
        row["dew_D6"],
        row["dew_D7"]
    ]) * units("degC")

    speed = np.array([
        row["windspeed_D1"],
        row["windspeed_D2"],
        row["windspeed_D3"],
        row["windspeed_D4"],
        row["windspeed_D5"],
        row["windspeed_D6"],
        row["windspeed_D7"]
    ]) * units("m/s")

    pres = np.array([
        row["sealevelpressure_D1"],
        row["sealevelpressure_D2"],
        row["sealevelpressure_D3"],
        row["sealevelpressure_D4"],
        row["sealevelpressure_D5"],
        row["sealevelpressure_D6"],
        row["sealevelpressure_D7"]
    ]) * units("hPa")

    solar = np.array([
        row["solarradiation_D1"],
        row["solarradiation_D2"],
        row["solarradiation_D3"],
        row["solarradiation_D4"],
        row["solarradiation_D5"],
        row["solarradiation_D6"],
        row["solarradiation_D7"]
    ]) * units("W/m^2")


    # --------------------------------------------------
    # Calculate WBGT
    # --------------------------------------------------

    result = wbgt(
        dates,
        latitude,
        longitude,
        solar,
        pres,
        temp_air,
        temp_dew,
        speed,
        method="liljegren"
    )


    # First result = WBGT
    wbgt_values = result[0].magnitude.round(2)

    all_wbgt.append(wbgt_values)


# --------------------------------------------------
# 4. Add WBGT columns to dataframe
# --------------------------------------------------

all_wbgt = np.array(all_wbgt)

df["WBGT_D1"] = all_wbgt[:, 0]
df["WBGT_D2"] = all_wbgt[:, 1]
df["WBGT_D3"] = all_wbgt[:, 2]
df["WBGT_D4"] = all_wbgt[:, 3]
df["WBGT_D5"] = all_wbgt[:, 4]
df["WBGT_D6"] = all_wbgt[:, 5]
df["WBGT_D7"] = all_wbgt[:, 6]


# --------------------------------------------------
# 5. Save output
# --------------------------------------------------

df.to_csv("weather_with_wbgt.csv", index=False)

print("WBGT calculation completed!")
print(df[[
    "name",
    "WBGT_D1",
    "WBGT_D2",
    "WBGT_D3",
    "WBGT_D4",
    "WBGT_D5",
    "WBGT_D6",
    "WBGT_D7"
]])

df.to_csv("wbgt-csv.csv",index=False)