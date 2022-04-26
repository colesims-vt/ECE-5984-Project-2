import os

import pandas as pd

# This is necessary to show lots of columns in pandas 0.12.
# Not necessary in pandas 0.13.
pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 60)

cwd = os.getcwd()

# Filepaths
tsla_file = cwd + '\\tsla_nasdaq_alltime.csv'
cpia_file = cwd + '\\CPIAUCSL.csv'
dff_file = cwd + '\\DFF.csv'
dj_file = cwd + '\\DOW JONES GLOBAL_DJUSAU.csv'

# Input Variables
begin_date = '3/1/2017'
end_date = '3/1/2022'

# Create days object from start and end dates
days = pd.date_range(start=begin_date, end=end_date, freq='D')


def format_dataset(filepath, date_label, interp=0, cols=[]):
    # Read in dataset
    df = pd.read_csv(filepath, parse_dates=[date_label], index_col=[date_label])
    # Filter by days of interest
    df = df.filter(items=days, axis=0)
    # If new labels are included, relabel columns
    if cols:
        df.columns = cols
    # If interp = 1, interpolate data
    if interp == 1:
        # Generate daily readings
        df = df.resample('D').mean()
        # Interpolate values
        for col in cols:
            df[col] = df[col].interpolate()

    return df


def clean_currency(x):
    """ If the value is a string, then remove currency symbol and delimiters
    otherwise, the value is numeric and can be converted
    """
    if isinstance(x, str):
        return (x.replace('$', '').replace(',', ''))
    return (x)


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

# Read in CPIAUCSL data
cpia = format_dataset(filepath=cpia_file, date_label='DATE', interp=1, cols=['CPIAUCSL'])

# Read in DFF data
dff = format_dataset(filepath=dff_file, date_label='DATE')

# Read in DJI data (Dow Jones)
dj = format_dataset(filepath=dj_file, date_label='Date', cols=['DJ Open','DJ High', 'DJ Low','DJ Close'])

print(dj.head())



output_data = tsla.join(cpia)
print(output_data.head())