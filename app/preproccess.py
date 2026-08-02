import pandas as pd 
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

# TODO : code must be a class or a function to input data from API 
# ----------------/for test/---------------- 
dataset_path = "./dataset/diabetes.csv" 
dataset_path = "./dataset/Start-Data-Analysis.xlsx"

if dataset_path.endswith('.xlsx'):
    dataset = pd.read_excel(dataset_path, sheet_name='Raw', header=1)
elif dataset_path.endswith('.csv'):
    dataset = pd.read_csv(dataset_path)
# ----------------/for test/---------------- 

unnamed_cols = dataset.columns[dataset.columns.str.contains('Unnamed:')].to_list()
empty_cols = lambda col: dataset[col].isnull().all()
remove_cols = [col for col in unnamed_cols if empty_cols(col)]
dataset = dataset.drop(columns=remove_cols)

duplicated = dataset.duplicated()

def find_uniques(col: pd.Series):
    return col.unique().shape[0]
uniques_percents = dataset.apply(find_uniques) / dataset.shape[0] * 100
print(uniques_percents)

dataset_summary = {
    'rows': dataset.shape[0],
    'columns': dataset.shape[1], 
    'number of duplicated rows': duplicated.sum(),
}

def find_unique_size(col: pd.Series):
    return col.unique().shape[0] == 1

def find_uniques(col: pd.Series):
    return col.unique() if col is not pd.Series().empty else 0

# potential numeric col not exact
def is_numeric(col: pd.Series):
    return is_numeric_dtype(col)

numeric_filter = dataset.apply(is_numeric)
numeric_filter = numeric_filter[numeric_filter].index
numeric_dataset = dataset[numeric_filter]

cardinality_thresh = 90
quality_report = {
    'missings': dataset.isnull().sum(), 
    'Constant Columns': dataset.T[dataset.apply(find_unique_size)].index,
    'Duplicate Rows': duplicated[duplicated], 
    f'High Cardinality(over than {cardinality_thresh}%)': uniques_percents[uniques_percents > cardinality_thresh]
}

describe = numeric_dataset.describe()
IQR = describe.loc['75%'] - describe.loc['25%']

numeric_columns_summary = {
    'describe': describe, 
    'IQR': IQR,
    'statistic MIN': describe.loc['25%'] - 1.5 * IQR,
    'statistic MAX': describe.loc['75%'] + 1.5 * IQR,
    'Skewness': numeric_dataset.skew(),
}
outlier_filter = (numeric_dataset > numeric_columns_summary['statistic MAX']) & (numeric_dataset < numeric_columns_summary['statistic MIN'])
numeric_columns_summary['Statistical Outliers (IQR)'] = numeric_dataset[outlier_filter]

# potential categorical col not exact
def is_categorical(col: pd.Series):
    unique_num = col.unique().__len__()
    all_num = col.__len__()
    unique_ratio = unique_num / all_num * 100 
    return unique_ratio <= 1  

categorical_prob_dataset = numeric_dataset.T[numeric_dataset.apply(is_categorical)].T

def find_top_value(col: pd.Series, just_top=True, dataset: pd.DataFrame | pd.Series = categorical_prob_dataset):
    if dataset is pd.DataFrame.empty:
        return None 
    elif type(dataset) == pd.DataFrame:
        col_name = col.name
        all_freq = dataset.groupby(col_name)[col_name].count().sort_values(ascending=False)
        if not all_freq.empty:
            return all_freq if not just_top else all_freq.iloc[0]
            
    else: 
        all_freq = dataset.groupby(dataset).count().sort_values(ascending=False)
        return all_freq if not just_top else all_freq.iloc[0]


categorical_columns_summary = {
    'Unique Count': categorical_prob_dataset.apply(find_uniques) if type(categorical_prob_dataset) == pd.DataFrame else categorical_columns_summary.unique(),
    'Top Values' : categorical_prob_dataset.apply(find_top_value),
    'Frequency': categorical_prob_dataset.apply(find_top_value, args=(False, categorical_prob_dataset))
}

def find_date_dtype(col):
    return is_datetime64_any_dtype(col)

date_filter = dataset.apply(find_date_dtype)
date_filter = date_filter[date_filter].index
date_dataset = dataset[date_filter]

date_column_summary = {
    'Min Date': date_dataset.min(),
    'Max Date': date_dataset.max(),
    'Missings': date_dataset[date_dataset.isnull()].count()
}

correlation = numeric_dataset.corr()

all_data_profile = {
    'Dataset Summary': dataset_summary,
    'Quality Report': quality_report,
    'Numeric Columns Summary': numeric_columns_summary,
    'Categorical Columns Summary': categorical_columns_summary,
    'Date Column Summary': date_column_summary,
    'Correlation': correlation
}

print(dataset_summary)
print(quality_report)
print(numeric_columns_summary)
print(categorical_columns_summary)
print(date_column_summary)
print(correlation)
print(all_data_profile)