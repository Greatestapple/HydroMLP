# Clone github data repository
!git clone https://github.com/Greatestapple/HydroMLP GitRepo

# set up working directory
import os
wdir = 'GitRepo/Data'
os.chdir(wdir)

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import random
import seaborn as sns

""" Data Preprocessing"""

# Read data
def load_ro_data(file_name):
    """
    Load and process Reverse Osmosis (RO) data from a CSV file.

    Returns:
        pd.DataFrame: Processed DataFrame with cleaned and formatted data
    """

    data_ro = pd.read_csv(file_name)

    data = pd.DataFrame(columns=['date','feed_pressure','feed_flow', 'turbidity','ph','temperature','toc','cleaned','dsc','dsr','stage3_flux','removal_efficiency'])


    data['date'] = pd.to_datetime(data_ro['date'],format="%Y-%m-%d")
    data['feed_pressure'] = pd.to_numeric(data_ro['feed_psi'],errors='ignore') # pressure in psi
    data['feed_flow'] = pd.to_numeric(data_ro['ff'],errors='ignore')
    data['turbidity'] = pd.to_numeric(data_ro['turb'],errors='ignore')
    data['ph'] = pd.to_numeric(data_ro['ph'],errors='ignore')
    data['temperature'] = pd.to_numeric(data_ro['temp'],errors='ignore') # units in F
    data['toc'] = pd.to_numeric(data_ro['rof_toc_avg'],errors='ignore')
    data['cleaned'] = pd.to_numeric(data_ro['cip'],errors='ignore') # 0=not cleaned, 1=cleaned
    data['dsc'] = pd.to_numeric(data_ro['dss'],errors='ignore') # days since cleaning
    data['dsr'] = pd.to_numeric(data_ro['days_since_replacement'],errors='ignore') # days since replacement
    data['stage3_flux'] = pd.to_numeric(data_ro['s3sf'],errors='ignore')
    data['removal_efficiency'] = pd.to_numeric(data_ro['unit_norm_percent_removal'],errors='ignore') # need to check

    # remove null values
    data = data.dropna()
    data = data.reset_index(drop=True)

    return data

data_a01 = load_ro_data('orange_county_ro_A01.csv')
data_a02 = load_ro_data('orange_county_ro_A02.csv')
data_a03 = load_ro_data('orange_county_ro_A03.csv')
data_b01 = load_ro_data('orange_county_ro_B01.csv')
data_b02 = load_ro_data('orange_county_ro_B02.csv')
data_b03 = load_ro_data('orange_county_ro_B03.csv')

raw_data_a01 = data_a01.copy()
raw_data_a02 = data_a02.copy()
raw_data_a03 = data_a03.copy()
raw_data_b01 = data_b01.copy()
raw_data_b02 = data_b02.copy()
raw_data_b03 = data_b03.copy()

print(data_a01)
# print(data_a02)
# print(data_a03)
# print(data_b01)
# print(data_b02)
# print(data_b03)

# visualize raw data
def visualize_data(data):

  fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(12, 21), sharex=True)

  axes[0].plot(data['date'], data['turbidity'], label='Turbidity', color='b')
  axes[0].set_ylabel('Turbidity (NTU)')
  axes[0].legend(loc='upper right')
  axes[0].set_title('Turbidity plot')

  axes[1].plot(data['date'], data['feed_pressure'], label='Pressure', color='b')
  axes[1].set_ylabel('Pressure')
  axes[1].legend(loc='upper right')
  axes[1].set_title('Pressure plot')

  axes[2].plot(data['date'], data['feed_flow'], label='Flow', color='b')
  axes[2].set_ylabel('Flow')
  axes[2].legend(loc='upper right')
  axes[2].set_title('Flow plot')

  axes[3].plot(data['date'], data['ph'], label='pH', color='b')
  axes[3].set_ylabel('pH')
  axes[3].legend(loc='upper right')
  axes[3].set_title('pH plot')

  axes[4].plot(data['date'], data['temperature'], label='Temperature', color='b')
  axes[4].set_ylabel('Temperature (°F)')
  axes[4].legend(loc='upper right')
  axes[4].set_title('Temperature plot')

  axes[5].plot(data['date'], data['toc'], label='TOC', color='b')
  axes[5].set_ylabel('TOC (mg/L)')
  axes[5].legend(loc='upper right')
  axes[5].set_title('TOC plot')

  axes[6].plot(data['date'], data['stage3_flux'], label='Stage 3 Flux', color='b')
  axes[6].set_ylabel('Flux')
  axes[6].legend(loc='upper right')
  axes[6].set_title('Stage 3 Flux plot')

  axes[7].plot(data['date'], data['removal_efficiency'], label='Removal Efficiency', color='b')
  axes[7].set_ylabel('Removal Efficiency')
  axes[7].legend(loc='upper right')
  axes[7].set_title('Removal Efficiency plot')

  # Common x-axis label
  axes[-1].set_xlabel('Date')

  # Improve layout
  plt.tight_layout()
  plt.show()

visualize_data(data_a01)
#visualize_data(data_a02)
#visualize_data(data_a03)
#visualize_data(data_b01)
#visualize_data(data_b02)
#visualize_data(data_b03)

# Data preprocessing
# use hampel to remove outliers
# use movemean to smooth datasets
def hampel_filter(data_series, window_size=5, n_sigmas=3):
    """
    Applies Hampel filter to detect and replace outliers.

    Parameters:
        data_series (pd.Series): Input data series
        window_size (int): Number of points before and after to consider
        n_sigmas (int): Threshold for defining outliers

    Returns:
        pd.Series: Cleaned series with outliers replaced
    """
    k = 1.4826  # Scale factor for standard deviation estimation
    rolling_median = data_series.rolling(window=2 * window_size, center=True).median()
    diff = np.abs(data_series - rolling_median)
    rolling_mad = k * diff.rolling(window=2 * window_size, center=True).median()

    # Define outliers
    outliers = diff > (n_sigmas * rolling_mad)

    # Replace outliers with rolling median
    data_series.loc[outliers] = rolling_median.loc[outliers]

    return data_series

window_size = 10  # Define smoothing window size

for col in ['turbidity', 'feed_pressure','feed_flow', 'ph', 'temperature', 'toc', 'stage3_flux','removal_efficiency']:
    data_a01[col] = hampel_filter(data_a01[col])  # Remove outliers
    data_a01[col] = data_a01[col].rolling(window=window_size, center=True).mean()  # Apply moving average
    data_a02[col] = hampel_filter(data_a02[col])  # Remove outliers
    data_a02[col] = data_a02[col].rolling(window=window_size, center=True).mean()  # Apply moving average
    data_a03[col] = hampel_filter(data_a03[col])  # Remove outliers
    data_a03[col] = data_a03[col].rolling(window=window_size, center=True).mean()  # Apply moving average
    data_b01[col] = hampel_filter(data_b01[col])  # Remove outliers
    data_b01[col] = data_b01[col].rolling(window=window_size, center=True).mean()  # Apply moving average
    data_b02[col] = hampel_filter(data_b02[col])  # Remove outliers
    data_b02[col] = data_b02[col].rolling(window=window_size, center=True).mean()  # Apply moving average
    data_b03[col] = hampel_filter(data_b03[col])  # Remove outliers
    data_b03[col] = data_b03[col].rolling(window=window_size, center=True).mean()  # Apply moving average

# # remove null values
# for data in [data_a01, data_a02, data_a03, data_b01, data_b02, data_b03]:
#   data = data.dropna()
#   data = data.reset_index(drop=True)

for i, df in enumerate([data_a01, data_a02, data_a03, data_b01, data_b02, data_b03]):
    first_10_rows = df.iloc[:10]   # Extract first 10 rows
    last_10_rows = df.iloc[-10:]   # Extract last 10 rows

    df = df.iloc[10:-10].reset_index(drop=True)  # Remove first and last 10 rows and reset index

    # Update the respective dataframe
    if i == 0:
        data_a01 = df
    elif i == 1:
        data_a02 = df
    elif i == 2:
        data_a03 = df
    elif i == 3:
        data_b01 = df
    elif i == 4:
        data_b02 = df
    elif i == 5:
        data_b03 = df

print(data_a01.head(20))

# Visualize preprocessed dataset

visualize_data(data_a01)
#visualize_data(data_a02)
#visualize_data(data_a03)
#visualize_data(data_b01)
#visualize_data(data_b02)
#visualize_data(data_b03)

# plots to compare raw data and preprocessed data
def compare_data(data, raw_data):
  fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(12, 21), sharex=True)

  axes[0].plot(data['date'], data['turbidity'], label='Preprocessed data', color='b')
  axes[0].plot(raw_data['date'], raw_data['turbidity'], label='Raw data', color='k',linestyle='dotted')
  axes[0].set_ylabel('Turbidity (NTU)')
  axes[0].legend(loc='upper right')
  axes[0].set_title('Comparison of Raw and Preprocessed Turbidity plot')

  axes[1].plot(data['date'], data['feed_pressure'], label='Preprocessed data', color='b')
  axes[1].plot(raw_data['date'], raw_data['feed_pressure'], label='Raw data', color='k',linestyle='dotted')
  axes[1].set_ylabel('Pressure')
  axes[1].legend(loc='upper right')
  axes[1].set_title('Comparison of Raw and Preprocessed Pressure plot')

  axes[2].plot(data['date'], data['feed_flow'], label='Preprocessed data', color='b')
  axes[2].plot(raw_data['date'], raw_data['feed_flow'], label='Raw data', color='k',linestyle='dotted')
  axes[2].set_ylabel('Flow')
  axes[2].legend(loc='upper right')
  axes[2].set_title('Comparison of Raw and Preprocessed Flow plot')

  axes[3].plot(data['date'], data['ph'], label='Preprocessed data', color='b')
  axes[3].plot(raw_data['date'], raw_data['ph'], label='Raw data', color='k',linestyle='dotted')
  axes[3].set_ylabel('pH')
  axes[3].legend(loc='upper right')
  axes[3].set_title('Comparison of Raw and Preprocessed pH plot')

  axes[4].plot(data['date'], data['temperature'], label='Preprocessed data', color='b')
  axes[4].plot(raw_data['date'], raw_data['temperature'], label='Raw data', color='k',linestyle='dotted')
  axes[4].set_ylabel('Temperature (°F)')
  axes[4].legend(loc='upper right')
  axes[4].set_title('Comparison of Raw and Preprocessed Temperature plot')

  axes[5].plot(data['date'], data['toc'], label='Preprocessed data', color='b')
  axes[5].plot(raw_data['date'], raw_data['toc'], label='Raw data', color='k',linestyle='dotted')
  axes[5].set_ylabel('TOC (mg/L)')
  axes[5].legend(loc='upper right')
  axes[5].set_title('Comparison of Raw and Preprocessed TOC plot')

  axes[6].plot(data['date'], data['stage3_flux'], label='Preprocessed data', color='b')
  axes[6].plot(raw_data['date'], raw_data['stage3_flux'], label='Raw data', color='k',linestyle='dotted')
  axes[6].set_ylabel('Flux')
  axes[6].legend(loc='upper right')
  axes[6].set_title('Comparison of Raw and Preprocessed Stage 3 Flux plot')

  axes[7].plot(data['date'], data['removal_efficiency'], label='Preprocessed data', color='b')
  axes[7].plot(raw_data['date'], raw_data['removal_efficiency'], label='Raw data', color='k',linestyle='dotted')
  axes[7].set_ylabel('Removal Efficiency')
  axes[7].legend(loc='upper right')
  axes[7].set_title('Comparison of Raw and Preprocessed Removal Efficiency plot')

  # Common x-axis label
  axes[-1].set_xlabel('Date')

  # Improve layout
  plt.tight_layout()
  plt.show()

compare_data(data_a01, raw_data_a01)
#compare_data(data_a02, raw_data_a02)
#compare_data(data_a03, raw_data_a03)
#compare_data(data_b01, raw_data_b01)
#compare_data(data_b02, raw_data_b02)
#compare_data(data_b03, raw_data_b03)

def compare_parameter(datasets, parameter):
    """
    Compare a specific parameter across multiple datasets.

    Parameters:
    - datasets (dict): A dictionary where keys are dataset names and values are pandas DataFrames.
    - parameter (str): The parameter to compare (e.g., 'turbidity', 'ph', etc.).

    Returns:
    - A plot comparing the parameter across datasets.
    """
    plt.figure(figsize=(10, 2))

    for name, df in datasets.items():
        plt.plot(df["date"], df[parameter], label=name, alpha=0.7)

    plt.xlabel("Date")
    plt.ylabel(parameter.capitalize())
    plt.title(f"Comparison of {parameter.capitalize()} Across Datasets")
    plt.legend()
    plt.show()

# Store datasets in a dictionary
datasets = {
    "data_a01": data_a01,
    "data_a02": data_a02,
    "data_a03": data_a03,
    "data_b01": data_b01,
    "data_b02": data_b02,
    "data_b03": data_b03
}


compare_parameter(datasets, "turbidity")
compare_parameter(datasets, "feed_pressure")
compare_parameter(datasets, "feed_flow")
compare_parameter(datasets, "ph")
compare_parameter(datasets, "temperature")
compare_parameter(datasets, "toc")
compare_parameter(datasets, "removal_efficiency")

"""# HydroMLP"""

print(data_a01)

X = data_a01[['turbidity', 'feed_pressure', 'feed_flow', 'ph', 'temperature', 'toc', 'cleaned', 'dsc', 'dsr']]
y = data_a01[['stage3_flux']]
# X = X.sort_values(by='datenum')
# y = y.loc[X.index]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Cumulative explained variance
cumulative_variance = np.cumsum(explained_variance_ratio)

# Find the number of components explaining at least 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1  # +1 because index starts at 0

print(f"Number of principal components needed to explain 95% variance: {n_components_95}")

# Apply PCA with the optimal number of components
pca_opt = PCA(n_components=n_components_95)
X_pca_opt = pca_opt.fit_transform(X_scaled)

print(f"Shape of transformed data with selected PCs: {X_pca_opt.shape}")

# Plot explained variance ratio and cumulative variance
fig, ax1 = plt.subplots(figsize=(10, 5))

# Bar plot for explained variance ratio
ax1.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, color='b', label='Explained Variance Ratio')
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Explained Variance Ratio', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Line plot for cumulative explained variance
ax2 = ax1.twinx()
ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='r', label='Cumulative Explained Variance')
ax2.axhline(y=0.95, color='r', linestyle='dashed', label="95% Threshold")
# ax2.axvline(x=n_components_95, color='purple', linestyle='dashed', label=f"{n_components_95} PCs")
ax2.set_ylabel('Cumulative Explained Variance', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Titles and labels
plt.title('Explained Variance Ratio and Cumulative Variance')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
fig.tight_layout()
plt.show()

# Get the absolute contributions of each input feature to the principal components
component_contributions = np.abs(pca_opt.components_).T  # Taking absolute values
feature_names = X.columns  # Feature names from DataFrame

# Plot contribution of each feature to the principal components
plt.figure(figsize=(7, 5))
plt.imshow(component_contributions, cmap='coolwarm', aspect='auto')
plt.colorbar(label="Contribution Magnitude")
plt.xticks(range(n_components_95), [f"PC{i+1}" for i in range(n_components_95)], rotation=45)
plt.yticks(range(len(feature_names)), feature_names)
plt.xlabel("Principal Components")
plt.ylabel("Original Features")
plt.title("Feature Contribution to Selected Principal Components")
plt.show()

# Convert data into a format suitable for Seaborn
contribution_df = pd.DataFrame(component_contributions, columns=[f"PC{i+1}" for i in range(n_components_95)])
contribution_df['Feature'] = feature_names

# Convert dataframe to long format for Seaborn
contribution_long = contribution_df.melt(id_vars='Feature', var_name='Principal Component', value_name='Contribution')

# Plot Violin Plot
plt.figure(figsize=(7, 5))
sns.violinplot(x='Principal Component', y='Contribution', data=contribution_long, inner="point", scale="width", palette="coolwarm")

# Labeling
plt.xticks(rotation=45)
plt.xlabel("Principal Components")
plt.ylabel("Feature Contribution Magnitude")
plt.title("Distribution of Feature Contributions Across Selected Principal Components")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Set random seeds
random.seed(1)  # Python built-in random seed
np.random.seed(1)  # NumPy random seed
tf.random.set_seed(1)  # TensorFlow random seed

# Define the MLP model with 2 outputs
def hydromlp(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(7, activation='relu', input_shape=(input_shape,)),
        keras.layers.Dense(25, activation='relu'),
        keras.layers.Dense(25, activation='relu'),
        keras.layers.Dense(40, activation='relu'),
        keras.layers.Dense(1)  # 1 outputs for 1 prediction
    ])

    # Compile model with mean_squared_error for multi-output regression
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
    return model

# Perform 5-fold K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)
rmse_scores = []
r2_scores = []

for train_index, test_index in kf.split(X_pca_opt):
    # Split randomly
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, shuffle=True)
    y_train, y_test = y_train.values, y_test.values

    # # Determine the split index
    # split_index = int(0.8 * len(X_pca_opt))

    # # Split the dataset sequentially
    # X_train, X_test = X_pca_opt[:split_index], X_pca_opt[split_index:]
    # y_train, y_test = y[:split_index], y[split_index:]

    # Initialize and train the model
    mlp_model = hydromlp(X_train.shape[1])
    mlp_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    # Evaluate the model
    y_pred = mlp_model.predict(X_test).flatten()

    rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2_scores.append(r2_score(y_test, y_pred))

# Compute mean and standard deviation across folds
mean_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)
mean_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)

# Display results
print(f"Cross-Validation Results:")
print(f"Average RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
print(f"Average R²: {mean_r2:.4f} ± {std_r2:.4f}")

# Plot RMSE per fold
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(rmse_scores) + 1), rmse_scores, marker='o', linestyle='-', color='b', label="RMSE per Fold")
plt.axhline(mean_rmse, color='r', linestyle='dashed', linewidth=2, label="Mean RMSE")
plt.xlabel("Fold")
plt.ylabel("RMSE")
plt.title("RMSE Across Cross-Validation Folds")
plt.legend()
plt.show()

# Plot R² per fold
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(r2_scores) + 1), r2_scores, marker='o', linestyle='-', color='g', label="R² Score per Fold")
plt.axhline(mean_r2, color='r', linestyle='dashed', linewidth=2, label="Mean R²")
plt.xlabel("Fold")
plt.ylabel("R² Score")
plt.title("R² Scores Across Cross-Validation Folds")
plt.legend()
plt.show()

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
r2_test = r2_score(y_test, y_pred)

# Print cross-validation results
print(f'Output - Average RMSE: {np.mean(rmse_test):.4f} ± {np.std(rmse_test):.4f}')
print(f'Output - Average R^2: {np.mean(r2_test):.4f} ± {np.std(r2_test):.4f}')

# # Plot predictions vs observations for last fold
# plt.figure(figsize=(12, 5))
# plt.scatter(y_test, y_pred, alpha=0.5)
# # Add a dashed 1:1 line (Perfect predictions)
# min_val = min(min(y_test), min(y_pred))
# max_val = max(max(y_test), max(y_pred))
# plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='dashed', linewidth=2, label="1:1 Line")
# #plt.plot(min(y_test), max(y_test), color='red', linestyle='dashed')
# plt.xlabel('Observed Output')
# plt.ylabel('Predicted Output')
# plt.title('Predictions vs Observations')

# plt.tight_layout()
# plt.show()

# Ensure y_test and y_pred are numeric
y_test = np.array(y_test, dtype=float).flatten()
y_pred = np.array(y_pred, dtype=float).flatten()

# Scatter plot of actual vs. predicted values
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='black')

# Annotate R² score on the plot
plt.text(min(y_test) + 0.05 * (max(y_test) - min(y_test)),
         min(y_pred) + 0.1 * (max(y_pred) - min(y_pred)),
         f"R2 = {r2_test:.2f}", fontsize=10)

# Labels and title
plt.xlabel("Observations")
plt.ylabel("Predictions")
plt.title("Observations vs. Predictions")

# Ensure y_test is a pandas Series with an index
if isinstance(y_test, np.ndarray):  # Convert if it's a NumPy array
    y_test = pd.Series(y_test.flatten(), index=pd.RangeIndex(start=0, stop=len(y_test), step=1))

if isinstance(y_pred, np.ndarray):  # Convert if it's a NumPy array
    y_pred = pd.Series(y_pred.flatten(), index=y_test.index)  # Use the same index as y_test

# Plot Predictions vs Observations Over Time (Test Set)
plt.figure(figsize=(6, 5))
plt.scatter(y_test.index, y_test, label="Observed", linestyle='-', marker='o', color='orange', alpha=0.6)
plt.scatter(y_test.index, y_pred, label="Predicted", linestyle='-', marker='o', color='blue', alpha=0.6)

# Labels and title
plt.xlabel('Index')
plt.ylabel('Output Value')
plt.title('Observations vs. Predictions')
plt.legend()
plt.show()