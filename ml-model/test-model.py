import pandas as pd
import joblib


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

MODEL_FILE = "random_forest_model.pkl"

model = joblib.load(MODEL_FILE)

print("Trained model loaded successfully.")


# ============================================================
# 2. CREATE SIMULATED DISTRICT DATA
# ============================================================
#
# Each row represents one hypothetical district.
#
# The values below are ONLY simulated examples.
# They are not real weather forecasts or mortality data.
# ============================================================

simulated_data = {

    "latitude": [
        27.2,
        28.6,
        25.4,
        26.8,
        29.0
    ],

    "longitude": [
        78.0,
        77.2,
        81.8,
        80.9,
        77.1
    ],

    "max_temp_5day": [
        39.0,
        42.0,
        44.0,
        46.0,
        38.0
    ],

    "mean_temp_5day": [
        36.5,
        39.5,
        41.0,
        43.0,
        35.5
    ],

    "max_humidity_5day": [
        55,
        60,
        65,
        70,
        80
    ],

    "mean_humidity_5day": [
        45,
        50,
        55,
        60,
        70
    ],

    "max_WBGT_5day": [
        27.5,
        29.5,
        31.0,
        33.0,
        30.5
    ],

    "mean_WBGT_5day": [
        26.0,
        28.0,
        29.5,
        31.0,
        28.5
    ],

    "dangerous_WBGT_days": [
        0,
        2,
        3,
        5,
        3
    ],

    "red_WBGT_days": [
        0,
        0,
        0,
        3,
        0
    ],

    "extreme_WBGT_days": [
        0,
        0,
        0,
        2,
        0
    ],

    "Total Main Workers": [
        50000,
        100000,
        150000,
        250000,
        80000
    ],

    "infant_pop": [
        30000,
        60000,
        90000,
        150000,
        50000
    ],

    "elderly_pop": [
        25000,
        50000,
        70000,
        110000,
        40000
    ]
}


# ============================================================
# 3. CONVERT TO DATAFRAME
# ============================================================

df = pd.DataFrame(simulated_data)


# ============================================================
# 4. PREDICT MORTALITY
# ============================================================

predictions = model.predict(df)


# ============================================================
# 5. ADD PREDICTION TO DATAFRAME
# ============================================================

df["Predicted_Excess_Deaths"] = predictions


# ============================================================
# 6. DISPLAY RESULTS
# ============================================================

print("\n==============================================")
print("SIMULATED HEATWAVE MORTALITY PREDICTIONS")
print("==============================================")

print(
    df[
        [
            "max_temp_5day",
            "mean_temp_5day",
            "max_humidity_5day",
            "max_WBGT_5day",
            "dangerous_WBGT_days",
            "red_WBGT_days",
            "extreme_WBGT_days",
            "infant_pop",
            "elderly_pop",
            "Predicted_Excess_Deaths"
        ]
    ].to_string(index=False)
)


# ============================================================
# 7. SAVE RESULTS
# ============================================================

df.to_csv(
    "simulated_predictions.csv",
    index=False
)

print("\nPrediction results saved to:")
print("simulated_predictions.csv")