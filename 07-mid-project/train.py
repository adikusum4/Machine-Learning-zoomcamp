# import library

import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error

##1st dataset for Air-quality by Counties of USA
data_air = 'aqi_yearly_1980_to_2021.csv'

df_air = pd.read_csv(data_air)

df_air.columns = df_air.columns.str.lower().str.replace(' ', '_')

# Create a new column by combining 'state' and 'county'
df_air['state_county'] = df_air['state'] + ' - ' + df_air['county']

##2nd dataset for Asthma prevalence by Counties of USA
data_asthma = '2020-PLACES__County_Data__GIS_Friendly_Format___2022_release_20241125.csv'

df_asthma = pd.read_csv(data_asthma)

# Displays columns containing the word "ASTHMA"
asthma_columns = df_asthma.filter(like='ASTHMA', axis=1).columns

# Select the relevant columns: StateDesc, CountyName, TotalPopulation, and asthma column.
df_asthma = df_asthma[['StateDesc', 'CountyName', 'TotalPopulation'] + asthma_columns.tolist()]

# Create a new column by combining 'StateDesc' and 'CountyName'
df_asthma['state_county'] = df_asthma['StateDesc'] + ' - ' + df_asthma['CountyName']

# Get unique values ​​of 'state_county' in both DataFrames
unique_air = df_air['state_county'].unique()
unique_asthma = df_asthma['state_county'].unique()

# Filter unique values ​​present in both DataFrames
common_state_county = set(unique_air).intersection(set(unique_asthma))

# Store the result in a new variable or DataFrame
common_state_county_list = list(common_state_county)

# Read data from CSV file
data_air = 'aqi_yearly_1980_to_2021.csv'
df_air = pd.read_csv(data_air)

# Filter data for years 2020, 2019, and 2018
df_air = df_air[df_air['Year'].isin([2020, 2019, 2018])]

df_air.columns = df_air.columns.str.lower().str.replace(' ', '_')

# Create a new column by combining 'state' and 'county'
df_air['state_county'] = df_air['state'] + ' - ' + df_air['county']

# Filter df_air for year 2020
df_air_2020 = df_air[df_air['year'] == 2020]

# Check if 'state_county' contains a value in common_state_county_list
df_air_2020_filtered = df_air_2020[df_air_2020['state_county'].isin(common_state_county_list)]

# Filter df_air for year 2019
df_air_2019 = df_air[df_air['year'] == 2019]

# Check if 'state_county' contains a value in common_state_county_list
df_air_2019_filtered = df_air_2019[df_air_2019['state_county'].isin(common_state_county_list)]

# Filter df_air for year 2018
df_air_2018 = df_air[df_air['year'] == 2018]

# Check if 'state_county' contains a value in common_state_county_list
df_air_2018_filtered = df_air_2018[df_air_2018['state_county'].isin(common_state_county_list)]

# Filter data for 2018, 2019 and 2020
df_air_2018_filtered = df_air_2018[df_air_2018['state_county'].isin(common_state_county_list)]
df_air_2019_filtered = df_air_2019[df_air_2019['state_county'].isin(common_state_county_list)]
df_air_2020_filtered = df_air_2020[df_air_2020['state_county'].isin(common_state_county_list)]

# Combine all filtered state_counties from 2018-2020
all_filtered_state_county = pd.concat([
    df_air_2018_filtered['state_county'],
    df_air_2019_filtered['state_county'],
    df_air_2020_filtered['state_county']
])

# Get state_county contained in all years
common_state_county_all_years = all_filtered_state_county.value_counts()

# Filter 'state_county' that appears 3 times
state_county_3_times = common_state_county_all_years[common_state_county_all_years == 3]

# Convert state_county that occurs 3 times to a list
state_county_3_list = state_county_3_times.index.tolist()

data_asthma_2020 = '2020-PLACES__County_Data__GIS_Friendly_Format___2022_release_20241125.csv'
df_asthma_2020 = pd.read_csv(data_asthma_2020)

data_asthma_2019 = '2019-PLACES__County_Data__GIS_Friendly_Format___2021_release_20241125.csv'
df_asthma_2019 = pd.read_csv(data_asthma_2019)

data_asthma_2018 = '2018-PLACES__County_Data__GIS_Friendly_Format___2020_release_20241125.csv'
df_asthma_2018 = pd.read_csv(data_asthma_2018)

# Displays columns containing the word "ASTHMA"
asthma_columns = df_asthma_2020.filter(like='ASTHMA', axis=1).columns

df_asthma_2020 = df_asthma_2020[['StateDesc', 'CountyName', 'TotalPopulation'] + asthma_columns.tolist()]
df_asthma_2019 = df_asthma_2019[['StateDesc', 'CountyName', 'TotalPopulation'] + asthma_columns.tolist()]
df_asthma_2018 = df_asthma_2018[['StateDesc', 'CountyName', 'TotalPopulation'] + asthma_columns.tolist()]

df_asthma_2020['state_county'] = df_asthma_2020['StateDesc'] + ' - ' + df_asthma_2020['CountyName']
df_asthma_2019['state_county'] = df_asthma_2019['StateDesc'] + ' - ' + df_asthma_2019['CountyName']
df_asthma_2018['state_county'] = df_asthma_2018['StateDesc'] + ' - ' + df_asthma_2018['CountyName']

# Filter df_air to only contain state_counties that are in state_county_3_list
df_air_filtered = df_air[df_air['state_county'].isin(state_county_3_list)]

df_asthma_2020_filtered = df_asthma_2020[df_asthma_2020['state_county'].isin(state_county_3_list)]
df_asthma_2019_filtered = df_asthma_2019[df_asthma_2019['state_county'].isin(state_county_3_list)]
df_asthma_2018_filtered = df_asthma_2018[df_asthma_2018['state_county'].isin(state_county_3_list)]

df_asthma_2019 = df_asthma_2019_filtered.dropna()

# Get unique values ​​from 'state_county' and convert them to a list
state_county_2019_list = df_asthma_2019['state_county'].unique().tolist()

df_asthma_2020 = df_asthma_2020[df_asthma_2020['state_county'].isin(state_county_2019_list)]
df_asthma_2018 = df_asthma_2018[df_asthma_2018['state_county'].isin(state_county_2019_list)]

# create column 'state_county_year' combining 'state_county' and year
df_asthma_2020['state_county_year'] = df_asthma_2020['state_county'] + ' - 2020'
df_asthma_2019['state_county_year'] = df_asthma_2019['state_county'] + ' - 2019'
df_asthma_2018['state_county_year'] = df_asthma_2018['state_county'] + ' - 2018'

# Combining DataFrames df_asthma_2018, df_asthma_2019, and df_asthma_2020
df_asthma = pd.concat([df_asthma_2018, df_asthma_2019, df_asthma_2020], ignore_index=True)

# Remove unnecessary columns
df_asthma = df_asthma.drop(columns=['StateDesc', 'CountyName', 'CASTHMA_CrudePrev',
                                    'CASTHMA_Crude95CI', 'CASTHMA_Adj95CI', 'state_county'])

# Filter df_air to only contain state_counties that are in state_county_3_list
df_air = df_air_filtered[df_air_filtered['state_county'].isin(state_county_2019_list)]

# Convert 'year' column to string and concatenate it with 'state_county'
df_air['state_county_year'] = df_air['state_county'] + ' - ' + df_air['year'].astype(str)

# EDA: Merge df_air and df_asthma based on 'state_county_year'
df_merged = pd.merge(df_air, df_asthma, on='state_county_year', how='inner')

df_merged = df_merged.drop(['state_county', 'state_county_year'], axis=1)

# Rename column 'TotalPopulation' to 'total_population'
df_merged = df_merged.rename(columns={'TotalPopulation': 'total_population'})

df_merged.columns = df_merged.columns.str.lower()

df_full_train, df_test = train_test_split(df_merged, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = np.log1p(df_train.casthma_adjprev.values)
y_val = np.log1p(df_val.casthma_adjprev.values)
y_test = np.log1p(df_test.casthma_adjprev.values)

del df_train['casthma_adjprev']
del df_val['casthma_adjprev']
del df_test['casthma_adjprev']

train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')

dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

fr = RandomForestRegressor(n_estimators=160, max_depth=45, 
                           random_state=42, n_jobs=-1)
fr.fit(X_train, y_train)
y_pred = fr.predict(X_val)
score = np.sqrt(mean_squared_error(y_val, y_pred))
print('score:')
print(score)

model_file = 'model1.bin'
dv_file = 'dv1.bin'

with open(model_file, 'wb') as f_out: 
    pickle.dump((fr), f_out)

with open(dv_file, 'wb') as f_out: 
    pickle.dump((dv), f_out)






