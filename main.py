# -*- coding: utf-8 -*-
"""Improved House Price Model"""

# ========== UPLOAD DATA ==========


import pandas as pd
import numpy as np

# Read the data
df = pd.read_csv("Chennai houseing sale.csv")

# ========== FEATURE ENGINEERING ==========
# Date fields
df['BUILD_AGE'] = 2025 - pd.to_datetime(df['DATE_BUILD'], dayfirst=True).dt.year
df['SALE_MONTH'] = pd.to_datetime(df['DATE_SALE'], dayfirst=True).dt.month

# Interaction feature
df['SQFT_x_QS_OVERALL'] = df['INT_SQFT'] * df['QS_OVERALL']

# Drop rows with missing target
df = df.dropna(subset=['SALES_PRICE'])

# One-hot encode AREA + other categorical features
df = pd.get_dummies(df, columns=['AREA', 'SALE_COND', 'BUILDTYPE', 'UTILITY_AVAIL', 'MZZONE'], drop_first=True)

# Features
feature_columns = [
    'INT_SQFT', 'DIST_MAINROAD', 'N_BEDROOM', 'N_BATHROOM', 'N_ROOM',
    'QS_ROOMS', 'QS_BATHROOM', 'QS_BEDROOM', 'QS_OVERALL',
    'BUILD_AGE', 'SALE_MONTH', 'SQFT_x_QS_OVERALL'
]
# Include all one-hot encoded columns
one_hot_cols = [col for col in df.columns if any(prefix in col for prefix in ['AREA_', 'SALE_COND_', 'BUILDTYPE_', 'UTILITY_AVAIL_', 'MZZONE_'])]
feature_columns.extend(one_hot_cols)

# Prepare training data
X = df[feature_columns]
y = np.log1p(df['SALES_PRICE'])  # log(price)

# ========== TRAIN MODEL ==========
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)

# ========== EVALUATION ==========
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)

mse = mean_squared_error(y_test_actual, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_actual, y_pred)

print(f"Mean Squared Error: ₹{mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): ₹{rmse:,.2f}")
print(f"R-squared: {r2:.4f}")

# ========== FEATURE IMPORTANCE ==========
import matplotlib.pyplot as plt

importance = model.feature_importances_
plt.figure(figsize=(12, max(6, len(feature_columns) * 0.3)))
plt.barh(feature_columns, importance)
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

# ========== NEW HOUSE PREDICTION ==========
# Example: Velachery home
new_house_raw = {
    'INT_SQFT': 1200,
    'DIST_MAINROAD': 5.5,
    'N_BEDROOM': 3,
    'N_BATHROOM': 2,
    'N_ROOM': 5,
    'QS_ROOMS': 7,
    'QS_BATHROOM': 7,
    'QS_BEDROOM': 7,
    'QS_OVERALL': 7,
    'AREA': 'Velachery',
    'DATE_BUILD': '15-05-2015',
    'DATE_SALE': '01-06-2024',
    'SALE_COND': 'Normal Sale',
    'BUILDTYPE': 'House',
    'UTILITY_AVAIL': 'AllPub',
    'MZZONE': 'RH',
}

new_df_raw = pd.DataFrame([new_house_raw])

# Date-based features
new_df_raw['BUILD_AGE'] = 2025 - pd.to_datetime(new_df_raw['DATE_BUILD'], format='%d-%m-%Y').dt.year
new_df_raw['SALE_MONTH'] = pd.to_datetime(new_df_raw['DATE_SALE'], format='%d-%m-%Y').dt.month
new_df_raw['SQFT_x_QS_OVERALL'] = new_df_raw['INT_SQFT'] * new_df_raw['QS_OVERALL']

# One-hot encoding
new_df = pd.get_dummies(new_df_raw, columns=['AREA', 'SALE_COND', 'BUILDTYPE', 'UTILITY_AVAIL', 'MZZONE'], drop_first=True)

# Match training feature columns
new_df = new_df.reindex(columns=X_train.columns, fill_value=0)

# Predict
predicted_log_price = model.predict(new_df)
predicted_price = np.expm1(predicted_log_price[0])  # reverse log1p

formatted_price = "₹{:,.2f}".format(predicted_price)
print(f"Predicted Price for New House: {formatted_price}")
