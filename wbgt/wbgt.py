import numpy as np
from pywbgt import wbgt
from metpy.units import units

# Example observation
dates = np.array([
    "2026-05-30",
    "2026-05-31",
    "2026-06-01",
    "2026-06-02",
    "2026-06-03",
    "2026-06-04",
    "2026-06-05"
], dtype="datetime64[D]")

latitudes = np.array([27.18])   # degrees North
longitudes = np.array([78.02])  # degrees East

# Solar radiation
solar = np.array([281.5]) * units("W/m^2")

# Atmospheric pressure
pres = np.array([996.5]) * units("hPa")

# Air temperature
temp_air = np.array([38]) * units("degC")

# Dew-point temperature
temp_dew = np.array([18.8]) * units("degC")

# Wind speed
speed = np.array([16.2]) * units("m/s")

# Calculate WBGT
result = wbgt(
    dates,
    latitudes,
    longitudes,
    solar,
    pres,
    temp_air,
    temp_dew,
    speed,
    method="liljegren"
)

print(result[0].magnitude.round(0).astype(int))

