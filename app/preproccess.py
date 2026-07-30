import pandas as pd 
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

dataset_path = "./dataset/diabetes.csv"
dataset = pd.read_csv(dataset_path)
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

numeric_filter = dataset.apply(is_numeric).index
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
    'Skewness': dataset.skew(),
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
        return all_freq if not just_top else all_freq.iloc[0]
    else: 
        all_freq = dataset.groupby(dataset).count().sort_values(ascending=False)
        return all_freq if not just_top else all_freq.iloc[0]


categorical_columns_summary = {
    'Unique Count': categorical_prob_dataset.apply(find_uniques) if type(categorical_prob_dataset) == pd.DataFrame else categorical_columns_summary.unique(),
    'Top Values' : categorical_prob_dataset.apply(find_top_value),
    'Frequency': categorical_prob_dataset.apply(find_top_value, args=(False, categorical_prob_dataset))
}

'''
Datetime Columns TODO 
│
├── Min Date
├── Max Date
└── Missing

Correlation Matrix TODO
integration TODO
'''
print(dataset_summary)
print(quality_report)
print(numeric_columns_summary)
print(categorical_columns_summary)