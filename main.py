import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

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
t10_file = cwd + '\\T10YIE.csv'
ted_file = cwd + '\\TEDRATE.csv'
unrate_file = cwd + '\\UNRATE.csv'
vix_file = cwd + '\\VIXCLS.csv'
wm2_file = cwd + '\\WM2NS.csv'

# Input Variables
begin_date = '3/1/2017'
begin_wide = '3/1/2016'
end_date = '3/1/2022'
end_wide = '3/1/2023'

# Create days object from start and end dates
days = pd.date_range(start=begin_date, end=end_date, freq='D')
wide_days = pd.date_range(start=begin_wide,end=end_wide, freq='D')


# Stats report class to generate DQR
class StatsReport:
    def __init__(self):
        self.stats_df = pd.DataFrame()
        self.stats_df['stat'] = ['cardinality', 'mean', 'median', 'n_at_median', 'mode', 'n_at_mode', 'stddev', 'min',
                                 'max', 'n_zero', 'n_missing']
        pass

    # Adds a column to the stats report
    def add_col(self, label, data):
        if pd.api.types.is_numeric_dtype(data):
            data_mean = data.mean()
            data_median = data.median()
            data_median_count = (data == data_median).sum()
            data_std = data.std()
            data_min = data.min()
            data_max = data.max()

        else:
            data_mean = ''
            data_median = ''
            data_median_count = ''
            data_std = ''
            data_min = ''
            data_max = ''
            # data_outliers = ''
        self.stats_df[label] = [data.nunique(), data_mean, data_median, data_median_count, data.mode()[0],
                                data.value_counts()[data.mode()[0]], data_std, data_min, data_max, (data == 0).sum(),
                                data.isna().sum()]

    # Prints stats report to string
    def to_string(self):
        return self.stats_df.to_string()

    # Saves stats report to excel
    def to_excel(self, name):
        return self.stats_df.to_excel(name)


# This function generates a DQR based on df, prints it to the console, and saves it to output_file
def stats_report(df, output_file):
    report = StatsReport()
    labels = df.columns
    # Create a simple data set summary for the console
    for thisLabel in labels:  # for each column, report basic stats
        this_col = df[thisLabel]
        report.add_col(thisLabel, this_col)

    report.to_excel(output_file)
    print(report.to_string())


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
    if interp == 1 or df.isna().any().any():
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
t10, output_data = format_dataset(filepath=t10_file, date_label='DATE', full_data=output_data)

# Read in TEDRATE
ted, output_data = format_dataset(filepath=ted_file, date_label='DATE', full_data=output_data)

# Read in UNRATE
unrate, output_data = format_dataset(filepath=unrate_file, date_label='DATE',interp=1, full_data=output_data)

# Read in VIXCLS
vix, output_data = format_dataset(filepath=vix_file, date_label='DATE', full_data=output_data)

# Read in WM2NS
wm2, output_data = format_dataset(filepath=wm2_file, date_label='DATE',interp=1, full_data=output_data)

# Create calculated fields
# Read in TSLA data
tsla_calcs = pd.read_csv(tsla_file, parse_dates=['Date'], index_col=['Date'])
# Format columns
for column in tsla_calcs.columns:
    if tsla_calcs[column].dtype=='object':
        tsla_calcs[column] = tsla_calcs[column].apply(clean_currency).astype('float')
# Create calculated fields dataframe and target dataframe
calc_fields = pd.DataFrame()
targets = pd.DataFrame()
# Add previous values to each row of TSLA data
for day_shift in range(1, 11):
    new_col = str(day_shift) + '_days_ago'
    shift_amt = -1 * day_shift
    calc_fields[new_col] = tsla_calcs['Close/Last'].shift(shift_amt)
# Add target value to TSLA data
targets['Target_Val'] = tsla_calcs['Close/Last'].shift(20)
targets['Target_Change'] = targets['Target_Val'] - tsla_calcs['Close/Last']
targets['Target_Pct_Change'] = targets['Target_Change']/tsla_calcs['Close/Last']
calc_fields['dy'] = tsla_calcs['Close/Last'] - calc_fields['1_days_ago']

# Create scaler
scaler = MinMaxScaler(feature_range=(-1, 1))
# Transform data
calc_fields[calc_fields.columns] = scaler.fit_transform(calc_fields[calc_fields.columns])

# Create normalized targets
target_scaler = MinMaxScaler(feature_range=(-1,1))
# Transform targets and store in new dataframe
targets_norm = targets.copy()
targets_norm[targets_norm.columns] = target_scaler.fit_transform(targets_norm[targets_norm.columns])

# Filter data by days of interest
targets = targets.filter(items=days, axis=0)
targets_norm = targets_norm.filter(items=days,axis=0)
calc_fields = calc_fields.filter(items=days, axis=0)

# Join the calculated fields to the output data
output_data = output_data.join(calc_fields)

# kNN imputation for missing data
imputer = KNNImputer()
output_data[output_data.columns] = imputer.fit_transform(output_data[output_data.columns])

# Run stats report on the data
# stats_report(output_data,'stats_report_predictors.xlsx')

x_train, x_test, y_train, y_test = train_test_split(output_data,targets,test_size=0.3, random_state=5000)

'''print(x_train.head())
print(y_train.head())'''

reg = LinearRegression().fit(x_train,y_train['Target_Val'])
y_pred = reg.predict(x_test)

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(y_train.index,y_train['Target_Val'],s=1,c='b',marker='s',label='training')
ax1.scatter(y_test.index,y_test['Target_Val'],s=1,c='k',marker='s',label='test')
ax1.scatter(y_test.index,y_pred,s=1,c='r',marker='s',label='predict')
plt.legend(loc='upper left')
plt.title('Linear Regression Model Results')
plt.show()


# Use MLPRegressor
regr = MLPRegressor(hidden_layer_sizes=(500,500,500),random_state=5000).fit(x_train,y_train['Target_Val'])
y_pred = regr.predict(x_test)

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(y_train.index,y_train['Target_Val'],s=1,c='b',marker='s',label='training')
ax1.scatter(y_test.index,y_test['Target_Val'],s=1,c='k',marker='s',label='test')
ax1.scatter(y_test.index,y_pred,s=1,c='r',marker='s',label='predict')
plt.legend(loc='upper left')
plt.title('Neural Network (MLPRegressor) Model Results')
plt.show()



'''print(output_data.head())
print(output_data.tail())

print(targets.head())
print(targets.tail())

print(targets_norm.head())
print(targets_norm.tail())'''