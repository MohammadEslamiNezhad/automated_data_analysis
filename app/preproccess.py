import pandas as pd 
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

cardinality_thresh = 90
quality_report = {
    'missings': dataset.isnull().sum(), 
    'Constant Columns': dataset.T[dataset.apply(find_unique_size)].index,
    'Duplicate Rows': duplicated[duplicated], 
    f'High Cardinality(over than {cardinality_thresh}%)': uniques_percents[uniques_percents > cardinality_thresh]
}

describe = dataset.describe()
describe.loc['min']
IQR = describe.loc['75%'] - describe.loc['25%']

numeric_columns_summary = {
    'describe': dataset.describe(), 
    'IQR': IQR,
    'statistic MIN': describe.loc['25%'] - 1.5 * IQR,
    'statistic MAX': describe.loc['75%'] + 1.5 * IQR,
    'Skewness': dataset.skew(),
    # 'Statistical Outliers (IQR)': [] TODO : i think this param can not show a good picture of data
}
outlier_filter = (dataset > numeric_columns_summary['statistic MAX']) & (dataset < numeric_columns_summary['statistic MIN'])
numeric_columns_summary['Statistical Outliers (IQR)'] = dataset[outlier_filter]


categorical_columns_summary = {
    
}

'''
Categorical Columns TODO 
│
├── Unique Count
├── Top Values
└── Frequency

Datetime Columns TODO 
│
├── Min Date
├── Max Date
└── Missing

Correlation Matrix TODO'''

print(dataset_summary)
print(quality_report)
print(numeric_columns_summary)
print(dataset.describe())
print(dataset.head())