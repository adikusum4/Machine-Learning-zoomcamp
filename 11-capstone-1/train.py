#!/usr/bin/env python
# coding: utf-8

# Predicting 2021 Residential Home Sale Prices in Durham, NC

# This code addresses the question: What is the value of a residential home in Durham, NC?

## Libraries for Data Manipulation and Profiling
import pandas as pd
import numpy as np

## Libraries for Machine Learning
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

## Libraries for Metrics
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score

## Libraries for Graphics
import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline

## Libraries for Serialization and Miscellaneous
import math
import pickle
import optuna

## Set Optuna logging level to WARNING to suppress logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")

# Import data
print("Import data...")
df = pd.read_csv('Durham_homes_sold_2021_YTD.csv')
print(df.info())


# Clean the Data
print()
print("Preparing the data...")
#print()
#print("standardize column names")
#print(df.columns)
#print("df.columns = df.columns.str.lower().str.replace(' ', '_')")
df.columns = df.columns.str.lower().str.replace(' ', '_')
#print(df.columns)
#print("column name has been standardized")

# Calculate the number of unique values for each column
nunique_values = df.nunique()

# Create a list of unique values for each column
unique_values = [', '.join(map(str, df[col].unique())) for col in df.columns]

# Calculate the count of NaN values for each column
count_nan = df.isna().sum()

# Calculate the percentage of NaN values for each column
nan_percentage = (count_nan / len(df)) * 100

# Get the data type for each column
dtype_values = df.dtypes

# Create a summary DataFrame with the desired information
df_summary = pd.DataFrame({
    'columns': df.columns,
    'nunique': nunique_values.values,
    'unique': unique_values,
    'dtype': dtype_values.values,
    'count_nan': count_nan.values,
    'nan_percentage': nan_percentage.values
})

#print()
#print("Displaying unique values, data types, and number of NaN values")
# Display the result
#print(df_summary)

#print()
#print("Displaying unique values, data types, and number of NaN values")
# Replace '$' and ',' in multiple columns
cols_to_replace = ['list_price', 'sold_price', 'total_living_area_sqft', 'approx_lot_sqft']

df[cols_to_replace] = df[cols_to_replace].replace({'\$': '', ',': ''}, regex=True)

# Convert columns to numeric type if necessary
df[cols_to_replace] = df[cols_to_replace].apply(pd.to_numeric, errors='coerce')

df['list_date'] = pd.to_datetime(df['list_date'])
df['closing_date'] = pd.to_datetime(df['closing_date'])

for c in ['list_date', 'closing_date']: 
    df[c] = pd.to_datetime(df[c])

df['list_year'] = df['list_date'].dt.year
df['year_built'] = df['year_built'].astype(int)
df['age_house'] = df['list_year'] - df['year_built']

df['days_on_market'] = (df['closing_date'] - df['list_date']).dt.days
df['closing_month'] = df['closing_date'].dt.month
df['closing_day'] = df['closing_date'].dt.day

# Calculate discount_price
df['discount_price'] = np.where(
    df['list_price'] == df['sold_price'],  # Condition: if list_price is equal to sold_price
    0,  # Value is 0 if the condition is met
    (df['list_price'] - df['sold_price']) / df['sold_price']  # Calculate the value if the condition is not met
)

del df['list_year']
del df['closing_date']
del df['list_date']

df['fireplace'] = df['fireplace'].replace({'4+':'4'})
df['fireplace'].astype(int)

df['zip'] = df['zip'].str[:5]
df['zip'].astype(int)

del df['city']

### Handle Missing Values

# Fill NaN values in 'hoa_1_fees_required' with values from 'hoa_y/n'
df['hoa_1_fees_required'].fillna(df['hoa_y/n'], inplace=True)

# Set 'hoa_1_fees_required' to 'No' if 'subdivision' is 'Not in a Subdivision'
df['hoa_1_fees_required'] = np.where(df['subdivision'] == 'Not in a Subdivision', 'No', df['hoa_1_fees_required'])

# Get a list of subdivisions with HOA fees as 'Yes' and 'No'
hoa_yes = df[df['hoa_1_fees_required'] == 'Yes']['subdivision'].unique()
hoa_no = df[df['hoa_1_fees_required'] == 'No']['subdivision'].unique()

# Fill NaN values based on conditions in 'hoa_1_fees_required' and 'subdivision'
df['hoa_1_fees_required'] = np.where(
    df['hoa_1_fees_required'].isna() & df['subdivision'].isin(hoa_yes),
    'Yes', df['hoa_1_fees_required']
)
df['hoa_1_fees_required'] = np.where(
    df['hoa_1_fees_required'].isna() & df['subdivision'].isin(hoa_no),
    'No', df['hoa_1_fees_required']
)

# If any NaN values remain, set 'No' as the default value
df['hoa_1_fees_required'].fillna('No', inplace=True)

del df['hoa_y/n']
del df['list_price']

df = df.fillna(-1)

df = df.replace('Yes', 1)
df = df.replace('No', 0)

categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

## Split the Data

df_copy = df.copy()

# Apply LabelEncoder to each categorical column
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df_copy[col].astype(str))  # Convert to string to avoid errors
    label_encoders[col] = le  # Store the LabelEncoder for reference or inverse transformation

df_full_train, df_test = train_test_split(df_copy, test_size=0.2, shuffle=False)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, shuffle=False)

df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = np.log1p(df_full_train.sold_price.values)
y_train = np.log1p(df_train.sold_price.values)
y_val = np.log1p(df_val.sold_price.values)
y_test = np.log1p(df_test.sold_price.values)

del df_full_train['sold_price']
del df_train['sold_price']
del df_val['sold_price']
del df_test['sold_price']

## Train the Models
print()
print("Training the model...")

dv = DictVectorizer(sparse=False)

train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)

# Objective function for CatBoost
def objective_catboost(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step=0.01),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 1, 255),
        'loss_function': 'RMSE',
        'random_state': 42
    }

    # Create a CatBoost model with the specified parameters
    model = CatBoostRegressor(**params, verbose=0)
    model.fit(X_train, y_train)

    # Predict on the validation data
    y_pred = model.predict(X_val)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse

# Run Optuna for CatBoost
study_catboost = optuna.create_study(direction='minimize')  # Minimize RMSE
study_catboost.optimize(objective_catboost, n_trials=50)

# Best results for CatBoost
print("Best parameters for CatBoost:")
print(study_catboost.best_trial.params)
print("Best RMSE for CatBoost: {:.5}".format(study_catboost.best_value))


## Validate the Model
print("Final model validated:")

best_params_catboost = study_catboost.best_trial.params
final_catboost = CatBoostRegressor(**best_params_catboost, verbose=0)
final_catboost.fit(X_train, y_train)

y_pred_catboost = final_catboost.predict(X_val)
final_rmse_catboost = np.sqrt(mean_squared_error(y_val, y_pred_catboost))
print("CatBoost R² score: {:.5}".format(final_catboost.score(X_val, y_val)))
print("Final RMSE for CatBoost: {:.5}".format(final_rmse_catboost))

## Save the Model
output_file = 'model_catboost.bin'
print()
print("Saving model as " + output_file)
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, final_catboost), f_out)


