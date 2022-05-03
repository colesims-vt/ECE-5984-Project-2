import os
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.impute import KNNImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, NuSVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# This is necessary to show lots of columns in pandas 0.12.
# Not necessary in pandas 0.13.
pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 60)

cwd = os.getcwd()

# Filepaths
tsla_file = cwd + '/tsla_nasdaq_alltime.csv'
cpia_file = cwd + '/CPIAUCSL.csv'
dff_file = cwd + '/DFF.csv'
dj_file = cwd + '/DJIA.csv'
gdp_file = cwd + '/GDP.csv'
lith_file = cwd + '/lithium_commodity_data.csv'
t10_file = cwd + '/T10YIE.csv'
ted_file = cwd + '/TEDRATE.csv'
unrate_file = cwd + '/UNRATE.csv'
vix_file = cwd + '/VIXCLS.csv'
wm2_file = cwd + '/WM2NS.csv'
tweet_file = cwd + '/tweets_Prepared.csv'
volume_file = cwd + '/tesla_volume_17-22.csv'
gasprice_file = cwd + '/weekly_gas_price.csv'

# Input Variables
begin_date = '3/1/2017'
begin_wide = '3/1/2016'
end_date = '3/1/2022'
end_wide = '3/1/2023'

# Create days object from start and end dates
days = pd.date_range(start=begin_date, end=end_date, freq='D')
wide_days = pd.date_range(start=begin_wide, end=end_wide, freq='D')

raw_dataframe = pd.DataFrame()
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


def format_dataset(filepath, date_label, interp=0, cols=[], full_data=pd.DataFrame(), raw_data=raw_dataframe):
    # Read in dataset
    df = pd.read_csv(filepath, parse_dates=[date_label], index_col=[date_label])
    raw_data = pd.concat([raw_data,df])
    # Get labels
    labels = df.columns
    # Replace periods
    df.replace(to_replace='.', inplace=True)
    # Remove any currency formatting
    for label in labels:
        if df[label].dtype == 'object':
            df[label] = df[label].apply(clean_currency).astype('float')
    # Create scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
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
    return df, full_data, raw_data


def clean_currency(x):
    """ If the value is a string, then remove currency symbol and delimiters
    otherwise, the value is numeric and can be converted
    """
    if isinstance(x, str):
        return (x.replace('$', '').replace(',', ''))
    return (x)


# Read feature data and add to dataset. Greyed out data negatively affected accuracy
# Read in TSLA data
tsla, output_data, raw_dataframe = format_dataset(filepath=tsla_file, date_label='Date',
                                   cols=['TSLA Close/Last', 'TSLA Volume', 'TSLA Open', 'TSLA High', 'TSLA Low'],raw_data=raw_dataframe)

# Read in CPIAUCSL data
cpia, output_data, raw_dataframe = format_dataset(filepath=cpia_file, date_label='DATE', interp=1, cols=['CPIAUCSL'],
                                   full_data=output_data,raw_data=raw_dataframe)

# Read in DFF data
dff, output_data, raw_dataframe = format_dataset(filepath=dff_file, date_label='DATE', full_data=output_data,raw_data=raw_dataframe)

# Read in DJI data (Dow Jones)
dj, output_data, raw_dataframe = format_dataset(filepath=dj_file, date_label='DATE', full_data=output_data,raw_data=raw_dataframe)

# Read in GDP data
gdp, output_data, raw_dataframe = format_dataset(filepath=gdp_file, date_label='DATE', interp=1, full_data=output_data,raw_data=raw_dataframe)

# Read in lithium data
lith, output_data, raw_dataframe = format_dataset(filepath=lith_file, date_label='Date',
                                   cols=['Lith Price', 'Lith Open', 'Lith High', 'Lith Low', 'Lith Pct Change'],
                                   full_data=output_data,raw_data=raw_dataframe)

# Read in T10YIE data
t10, output_data, raw_dataframe = format_dataset(filepath=t10_file, date_label='DATE', full_data=output_data,raw_data=raw_dataframe)

# Read in TEDRATE
ted, output_data, raw_dataframe = format_dataset(filepath=ted_file, date_label='DATE', full_data=output_data,raw_data=raw_dataframe)

# Read in UNRATE
unrate, output_data, raw_dataframe = format_dataset(filepath=unrate_file, date_label='DATE', interp=1, full_data=output_data,raw_data=raw_dataframe)

# Read in VIXCLS
vix, output_data, raw_dataframe = format_dataset(filepath=vix_file, date_label='DATE', full_data=output_data,raw_data=raw_dataframe)

# Read in WM2NS
wm2, output_data, raw_dataframe = format_dataset(filepath=wm2_file, date_label='DATE', interp=1, full_data=output_data,raw_data=raw_dataframe)

# Read in Tweet Data
# tweet, output_data, raw_dataframe = format_dataset(filepath=tweet_file, date_label='date', interp=1, full_data=output_data,raw_data=raw_dataframe)

# Read in Tesla Volume Data
volume, output_data, raw_dataframe = format_dataset(filepath=volume_file, date_label='Date', interp=1, full_data=output_data,raw_data=raw_dataframe)

# Read in Weekly Gas Price Data
gas, output_data, raw_dataframe = format_dataset(filepath=gasprice_file, date_label='date', interp=1, full_data=output_data,raw_data=raw_dataframe)

# Create calculated fields
# Read in TSLA data
tsla_calcs = pd.read_csv(tsla_file, parse_dates=['Date'], index_col=['Date'])
# Format columns
for column in tsla_calcs.columns:
    if tsla_calcs[column].dtype == 'object':
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
targets['Target_Pct_Change'] = targets['Target_Change'] / tsla_calcs['Close/Last']
calc_fields['dy'] = tsla_calcs['Close/Last'] - calc_fields['1_days_ago']
targets['direction'] = np.sign(targets['Target_Change'])

# Create scaler
scaler = MinMaxScaler(feature_range=(-1, 1))
# Transform data
calc_fields[calc_fields.columns] = scaler.fit_transform(calc_fields[calc_fields.columns])

# Create normalized targets
target_scaler = MinMaxScaler(feature_range=(-1, 1))
# Transform targets and store in new dataframe
targets_norm = targets.copy()
targets_norm[targets_norm.columns] = target_scaler.fit_transform(targets_norm[targets_norm.columns])

# Filter data by days of interest
targets = targets.filter(items=days, axis=0)
targets_norm = targets_norm.filter(items=days, axis=0)
calc_fields = calc_fields.filter(items=days, axis=0)
tsla_calcs = tsla_calcs.filter(items=days, axis=0)

# Join the calculated fields to the output data
output_data = output_data.join(calc_fields)

# kNN imputation for missing data
imputer = KNNImputer()
output_data[output_data.columns] = imputer.fit_transform(output_data[output_data.columns])

# Run stats report on the data
stats_report(output_data,'stats_report_processed_data.xlsx')
stats_report(raw_dataframe,'stats_report_raw_data.xlsx')

x_train, x_test, y_train, y_test = train_test_split(output_data, targets, test_size=0.3, random_state=5000)

'''print(x_train.head())
print(y_train.head())'''

'''# Run linear regression
reg = LinearRegression().fit(x_train,y_train['Target_Val'])
y_pred = reg.predict(x_test)

# Calculate MSE and R2
y_correct = y_test['Target_Val'].to_numpy()
linear_mse = sklearn.metrics.mean_squared_error(y_correct, y_pred)
print('Linear MSE in Normalized Units', linear_mse, '\n')
linear_r2 = sklearn.metrics.r2_score(y_correct, y_pred)
print('Linear R-Squared in Normalized Units', linear_r2, '\n')

# plot linear regression
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(y_train.index,y_train['Target_Val'],s=1,c='b',marker='s',label='training')
ax1.scatter(y_test.index,y_test['Target_Val'],s=1,c='k',marker='s',label='test')
ax1.scatter(y_test.index,y_pred,s=1,c='r',marker='s',label='predict')
plt.legend(loc='upper left')
plt.title('Linear Regression Model Results')
plt.show()


# Use MLPRegressor
regr = MLPRegressor(hidden_layer_sizes=(50,50,50,50,50,50),random_state=5000,max_iter=10000).fit(x_train,y_train['Target_Val'])
y_pred = regr.predict(x_test)

for i in range(0,len(x_train.columns)):
    print(x_train.columns[i],reg.coef_[i])

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(y_train.index,y_train['Target_Val'],s=1,c='b',marker='s',label='training')
ax1.scatter(y_test.index,y_test['Target_Val'],s=1,c='k',marker='s',label='test')
ax1.scatter(y_test.index,y_pred,s=1,c='r',marker='s',label='predict')
plt.legend(loc='upper left')
plt.title('Neural Network (MLPRegressor) Model Results')
plt.show()'''
# Train All Classifiers
# Split the Data
x_train, x_test, y_train, y_test = train_test_split(output_data, targets['direction'], test_size=0.3, random_state=5000)

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100), random_state=5000),
    LogisticRegression()
]

# Logging for Visual Comparison
log_cols = ["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

# Track name and score
clf_names = []
clf_scores = []

for clf in classifiers:
    clf.fit(x_train, y_train)
    name = clf.__class__.__name__

    print("=" * 30)
    print(name)
    clf_names.append(name)

    print('****Results****')
    train_predictions = clf.predict(x_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    clf_scores.append(acc)

    train_predictions = clf.predict_proba(x_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))

    log_entry = pd.DataFrame([[name, acc * 100, ll]], columns=log_cols)
    log = log.append(log_entry)

print("=" * 30)

d = {'clf_name': clf_names, 'clf_score': clf_scores}
clf_data = pd.DataFrame(data=d)
clf_data.to_csv('classifier_data.csv',index=False)

# Train all Regressors
# Split the Data
x_train, x_test, y_train, y_test = train_test_split(output_data, targets['Target_Val'], test_size=0.3,
                                                    random_state=5000)

regr_names = []
regr_scores = []

regressors = [
    LinearRegression(),
    LGBMRegressor(),
    XGBRegressor(verbosity=0),
    CatBoostRegressor(silent=True),
    SGDRegressor(),
    KernelRidge(),
    ElasticNet(),
    BayesianRidge(),
    GradientBoostingRegressor(),
    SVR(),
    MLPRegressor(hidden_layer_sizes=(100,50,25),max_iter=10000)
]

# Logging for Visual Output
log_cols = ['Regressor', 'MSE']

for regr in regressors:
    regr.fit(x_train, y_train)
    name = regr.__class__.__name__

    print('=' * 30)
    print(name)
    regr_names.append(name)

    print('****Results****')
    train_predictions = regr.predict(x_test)
    mse = mean_squared_error(y_test, train_predictions)
    print('MSE:', mse)
    regr_scores.append(mse)

print('=' * 30)

d = {'regr_name': regr_names, 'regr_score': regr_scores}
regr_data = pd.DataFrame(data=d)
regr_data.to_csv('regression_data.csv',index=False)

# Multi-Stage
# 1. Classifier for Price Direction
# 2. Regressor for Price Change
x_train, x_test, y_train, y_test = train_test_split(output_data, targets, test_size=0.3, random_state=5000)
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x_train, y_train['direction'], test_size=0.3,
                                                            random_state=5000)
clf = RandomForestClassifier().fit(x_train_1, y_train_1)
stage_1_score = clf.score(x_test_1, y_test_1)
print('Stage 1 Score: ', stage_1_score)

x_train['direc'] = clf.predict(x_train)

x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x_train, y_train['Target_Val'], test_size=0.3,
                                                            random_state=5000)

regr2 = CatBoostRegressor(depth=4, learning_rate=0.15, silent=True).fit(x_train_2, y_train_2)

y_pred = regr2.predict(x_test_2)

stage_2_mse = mean_squared_error(y_test_2, y_pred)

print('Stage 2 MSE: ', stage_2_mse)

x_test['direc'] = clf.predict(x_test)
y_pred_full = regr2.predict(x_test)
full_mse = mean_squared_error(y_test['Target_Val'], y_pred_full)

print('Multi-Stage MSE: ', full_mse)

# Choose Best Model for Output (this is Single Stage CatBoost)
'''x_train, x_test, y_train, y_test = train_test_split(output_data, targets['Target_Val'], test_size=0.3, random_state=5000)

model_CBR = CatBoostRegressor(silent=True)

parameters = {'depth': [2, 4, 6],
              'iterations': [100, 500, 1000],
              'learning_rate': [0.1, 0.15, 0.2],
              }

grid = GridSearchCV(estimator=model_CBR, param_grid=parameters, cv=2, n_jobs=-1)
grid.fit(x_train,y_train)

print(" Results from Grid Search ")
print("\n The best estimator across ALL searched params:\n", grid.best_estimator_)
print("\n The best score across ALL searched params:\n", grid.best_score_)
print("\n The best parameters across ALL searched params:\n", grid.best_params_)'''

x_train, x_test, y_train, y_test = train_test_split(output_data, targets['Target_Val'], test_size=0.3, random_state=9000)

regr = CatBoostRegressor(depth=4, learning_rate=0.15, silent=True).fit(x_train,y_train)
y_pred = regr.predict(x_test)
mse = mean_squared_error(y_test,y_pred)

print('Final MSE:',mse)


full_predict = regr.predict(output_data)
d = {'current':tsla_calcs['Close/Last'],'pred':full_predict,'actual':targets['Target_Val']}
df = pd.DataFrame(data=d,index=output_data.index)
df.to_csv('price_predict.csv')

thresholds = [-0.05, -0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05]
returns = []
buys = []
for thresh in thresholds:
    total_returns = 0
    number_buys = 0
    for y in range(1, len(df)):
        pred = df['pred'].iloc[y]
        current = df['current'].iloc[y]
        actual = df['actual'].iloc[y]

        if (pred - current)/current > thresh:
            total_returns += actual/current * 1000 - 1000
            number_buys += 1
    returns.append(total_returns)
    buys.append(number_buys)

pos_returns = 0
pos_buys = 0
neg_returns = 0
neg_buys = 0
every_returns = 0
every_buys = 0
for y in range(2, len(df)):
    current = df['current'].iloc[y]
    prev = df['current'].iloc[y-1]
    actual = df['actual'].iloc[y]

    if current > prev:
        pos_returns += actual/current * 1000 - 1000
        pos_buys += 1
    if current < prev:
        neg_returns += actual/current * 1000 - 1000
        neg_buys += 1
    every_returns += actual/current * 1000 - 1000
    every_buys += 1

thresholds.append('buypos')
thresholds.append('buyneg')
thresholds.append('buyevery')
returns.append(pos_returns)
returns.append(neg_returns)
returns.append(every_returns)
buys.append(pos_buys)
buys.append(neg_buys)
buys.append(every_buys)



d = {'thresholds':thresholds,'returns':returns,'buys':buys}
df = pd.DataFrame(data=d)
df.to_csv('invest_results.csv',index=False)


