import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# This is necessary to show lots of columns in pandas 0.12.
# Not necessary in pandas 0.13.
pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 60)

cwd = os.getcwd()

# Filepaths
tsla_file = cwd + '\\tsla_nasdaq_alltime.csv'
cpia_file = cwd + '\\CPIAUCSL.csv'
dff_file = cwd + '\\DFF.csv'
dj_file = cwd + '\\DJIA.csv'
gdp_file = cwd + '\\GDP.csv'
lith_file = cwd + '\\lithium_commodity_data.csv'
T10_file = cwd + '\\T10YIE.csv'

# Input Variables
begin_date = '3/1/2017'
begin_wide = '3/1/2016'
end_date = '3/1/2022'
end_wide = '3/1/2023'

# Create days object from start and end dates
days = pd.date_range(start=begin_date, end=end_date, freq='D')
wide_days = pd.date_range(start=begin_wide,end=end_wide, freq='D')


def format_dataset(filepath, date_label, interp=0, cols=[], full_data=pd.DataFrame()):
    # Read in dataset
    df = pd.read_csv(filepath, parse_dates=[date_label], index_col=[date_label])
    # Get labels
    labels = df.columns
    # Replace periods
    df.replace(to_replace='.',inplace=True)
    # Remove any currency formatting
    for label in labels:
        if df[label].dtype == 'object':
            df[label] = df[label].apply(clean_currency).astype('float')
    # Create scaler
    scaler = MinMaxScaler(feature_range=(-1,1))
    # Transform data
    df[labels] = scaler.fit_transform(df[labels])
    # Filter by days of interest
    df = df.filter(items=wide_days, axis=0)
    # If new labels are included, relabel columns
    if cols:
        df.columns = cols
    else:
        cols = df.columns
    # If interp = 1, interpolate data
    if interp == 1:
        # Generate daily readings
        df = df.resample('D').mean()
        # Interpolate values
        for col in cols:
            df[col] = df[col].interpolate()
    df = df.filter(items=days, axis=0)
    if not full_data.empty:
        full_data = full_data.join(df)
    else:
        full_data = df
    return df, full_data


def clean_currency(x):
    """ If the value is a string, then remove currency symbol and delimiters
    otherwise, the value is numeric and can be converted
    """
    if isinstance(x, str):
        return (x.replace('$', '').replace(',', ''))
    return (x)

# Read in TSLA data
tsla, output_data = format_dataset(filepath=tsla_file,date_label='Date',cols=['TSLA Close/Last','TSLA Volume','TSLA Open','TSLA High','TSLA Low'])

# Read in CPIAUCSL data
cpia, output_data = format_dataset(filepath=cpia_file, date_label='DATE', interp=1, cols=['CPIAUCSL'], full_data=output_data)

# Read in DFF data
dff, output_data = format_dataset(filepath=dff_file, date_label='DATE', full_data=output_data)

# Read in DJI data (Dow Jones)
dj, output_data = format_dataset(filepath=dj_file, date_label='DATE', full_data=output_data)

# Read in GDP data
gdp, output_data = format_dataset(filepath=gdp_file, date_label='DATE',interp=1, full_data=output_data)

# Read in lithium data
lith, output_data = format_dataset(filepath=lith_file, date_label='Date', cols=['Lith Price','Lith Open','Lith High','Lith Low','Lith Pct Change'], full_data=output_data)

# Read in T10YIE data
t10, output_data = format_dataset(filepath=T10_file, date_label='DATE', full_data=output_data)


print(output_data.head())
print(output_data.tail())


'''
# Read in TSLA data
tsla = pd.read_csv(tsla_file, parse_dates=['Date'], index_col=['Date'])
# Add previous values to each row of TSLA data
for day_shift in range(1, 11):
    new_col = str(day_shift) + '_days_ago'
    shift_amt = -1 * day_shift
    tsla[new_col] = tsla['Close/Last'].shift(shift_amt)
# Add target value to TSLA data
tsla['Target_Val'] = tsla['Close/Last'].shift(20)
# Filter TSLA data by days of interest
tsla = tsla.filter(items=days, axis=0)
# Format currency columns as float
for column in tsla.columns:
    if tsla[column].dtype=='object':
        tsla[column] = tsla[column].apply(clean_currency).astype('float')
# Create additional calculated target columns
tsla['Target_Change'] = tsla['Target_Val'] - tsla['Close/Last']
tsla['Target_Pct_Change'] = tsla['Target_Change']/tsla['Close/Last']
'''