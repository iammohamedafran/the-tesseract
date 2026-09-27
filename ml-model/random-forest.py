import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. SETTINGS
# ============================================================

INPUT_FILE = "super-da-thambi-modified.csv"

TARGET = "Total Excess deaths due to a 5-day heatwave (97th percentile)"

RANDOM_STATE = 42


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# 3. FEATURES
# ============================================================

features = [

    # Location
    "latitude",
    "longitude",

    # 5-day temperature
    "max_temp_5day",
    "mean_temp_5day",

    # 5-day humidity
    "max_humidity_5day",
    "mean_humidity_5day",

    # 5-day WBGT
    "max_WBGT_5day",
    "mean_WBGT_5day",

    # WBGT danger
    "dangerous_WBGT_days",
    "red_WBGT_days",
    "extreme_WBGT_days",

    # Population / exposure
    "infant_density",
    "elderly_density",
    "worker_density",
]


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = features + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    print("\nERROR: The following columns are missing:")

    for col in missing_columns:
        print(" -", col)

    raise SystemExit


# ============================================================
# 5. SELECT FEATURES AND TARGET
# ============================================================

X = df[features].copy()
y = df[TARGET].copy()


# ============================================================
# 6. HANDLE MISSING VALUES
# ============================================================

print("\nMissing values before cleaning:")
print(X.isnull().sum())

# Fill feature missing values using median
X = X.fillna(X.median())

# Remove rows where target is missing
valid_rows = y.notna()

X = X.loc[valid_rows]
y = y.loc[valid_rows]


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 8. DEFINE MODELS
# ============================================================

models = {

    # --------------------------------------------------------
    # Ridge Regression
    # --------------------------------------------------------

    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10.0))
    ]),


    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),


    # --------------------------------------------------------
    # Gradient Boosting
    # --------------------------------------------------------

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        loss="huber",
        random_state=RANDOM_STATE
    )
}


# ============================================================
# 9. TRAIN AND EVALUATE ALL MODELS
# ============================================================

results = []

trained_models = {}

print("\n==============================================")
print("MODEL TRAINING")
print("==============================================")


for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train on training data
    model.fit(X_train, y_train)

    # Predict test data
    predictions = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # Store results
    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[name] = model

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.3f}")


# ============================================================
# 10. OPTIONAL XGBOOST
# ============================================================

try:

    from xgboost import XGBRegressor

    print("\nTraining XGBoost...")

    xgb_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE
    )

    xgb_model.fit(
        X_train,
        y_train
    )

    predictions = xgb_model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": "XGBoost",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models["XGBoost"] = xgb_model

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.3f}")

except ImportError:

    print("\nXGBoost is not installed.")
    print("Skipping XGBoost.")


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print("\n==============================================")
print("MODEL COMPARISON")
print("==============================================")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


# ============================================================
# 12. 5-FOLD CROSS VALIDATION
# ============================================================

print("\n==============================================")
print("5-FOLD CROSS VALIDATION")
print("==============================================")

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

cv_results = []


for name, model in trained_models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_mean_absolute_error"
    )

    mae_scores = -scores

    mean_mae = mae_scores.mean()
    std_mae = mae_scores.std()

    cv_results.append({
        "Model": name,
        "CV MAE Mean": mean_mae,
        "CV MAE Std": std_mae
    })

    print(
        f"{name}: "
        f"MAE = {mean_mae:.2f} "
        f"+/- {std_mae:.2f}"
    )


# ============================================================
# 13. TRAIN ALL MODELS ON COMPLETE DATASET
# ============================================================

print("\n==============================================")
print("TRAINING FINAL MODELS")
print("==============================================")

for name, model in trained_models.items():

    print(f"Training final {name}...")

    model.fit(
        X,
        y
    )


# ============================================================
# 14. SAVE ALL THREE MAIN MODELS
# ============================================================

print("\n==============================================")
print("SAVING MODELS")
print("==============================================")


# Ridge
joblib.dump(
    trained_models["Ridge Regression"],
    "ridge_model.pkl"
)

print("Saved: ridge_model.pkl")


# Random Forest
joblib.dump(
    trained_models["Random Forest"],
    "random_forest_model.pkl"
)

print("Saved: random_forest_model.pkl")


# Gradient Boosting
joblib.dump(
    trained_models["Gradient Boosting"],
    "gradient_boosting_model.pkl"
)

print("Saved: gradient_boosting_model.pkl")


# Optional XGBoost
if "XGBoost" in trained_models:

    joblib.dump(
        trained_models["XGBoost"],
        "xgboost_model.pkl"
    )

    print("Saved: xgboost_model.pkl")


# ============================================================
# 15. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

rf = trained_models["Random Forest"]

importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\n==============================================")
print("RANDOM FOREST FEATURE IMPORTANCE")
print("==============================================")

print(
    importance.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

importance.to_csv(
    "feature_importance_forest.csv",
    index=False
)
# ============================================================
# 15. GRADIENT BOOSTING FEATURE IMPORTANCE
# ============================================================

gb = trained_models["Gradient Boosting"]

importance = pd.DataFrame({
    "Feature": features,
    "Importance": gb.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\n==============================================")
print("GRADIENT BOOSTING FEATURE IMPORTANCE")
print("==============================================")

print(
    importance.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

importance.to_csv(
    "gradient_boosting_feature_importance.csv",
    index=False
)


# ============================================================
# 16. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

cv_df = pd.DataFrame(cv_results)

cv_df.to_csv(
    "cross_validation_results.csv",
    index=False
)


# ============================================================
# 17. FINISHED
# ============================================================

print("\n==============================================")
print("TRAINING COMPLETED")
print("==============================================")

print("\nFiles created:")

print(" - ridge_model.pkl")
print(" - random_forest_model.pkl")
print(" - gradient_boosting_model.pkl")

if "XGBoost" in trained_models:
    print(" - xgboost_model.pkl")

print(" - feature_importance.csv")
print(" - model_comparison.csv")
print(" - cross_validation_results.csv")