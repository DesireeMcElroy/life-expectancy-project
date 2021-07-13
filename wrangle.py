import pandas as pd
import numpy as np

import os

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def acquire_who_data():

    # Read fresh data from db into a DataFrame
    df = pd.read_csv('Life Expectancy Data.csv')


    return df


def get_info(df):
    '''
    This function takes in a dataframe and prints out information about the dataframe.
    '''

    print(df.info())
    print()
    print('------------------------')
    print()
    print('This dataframe has', df.shape[0], 'rows and', df.shape[1], 'columns.')
    print()
    print('------------------------')
    print()
    print('Null count in dataframe:')
    print('------------------------')
    print(df.isnull().sum())
    print()
    print('------------------------')
    print(' Dataframe sample:')
    print()
    return df.sample(3)


def prep_who(df):
    
    # strip leading whitespaces in columns
    df.columns = df.columns.str.strip()
    
    # lower case the columns
    df.rename(str.lower, axis='columns', inplace=True)
    
    # rename columns
    df.rename(columns={'status': 'developed_country',
                       'life expectancy':'life_expectancy',
                       'adult mortality': 'adult_mortality',
                       'infant deaths': 'infant_deaths',
                       'percentage expenditure': 'pct_expenditure',
                       'hepatitis b': 'hep_b',
                       'under-five deaths': 'under_five_deaths',
                       'total expenditure': 'total_expenditure',
                       'thinness  1-19 years': 'thinness_10-19yrs',
                       'thinness 5-9 years': 'thinness_5-9yrs',
                       'income composition of resources': 'income_comp_resources',
                       'schooling': 'yrs_education'
    }, inplace=True)
    
    # hot encode status column into a bool
    df['developed_country'] = np.where(df['developed_country'] == 'Developed', 1, 0)
    
    return df


def handle_who_nulls(df):
    '''
    This function takes in my who life expectancy dataframe and 
    '''

    # fill null for hep_b column using mean average
    df['country_avg'] = df.groupby('country').hep_b.transform('mean')
    df.hep_b = df.hep_b.fillna(df.country_avg)


    # fill alcohol na columns with last known alcohol entry since it is a linear trend per country
    df['country_avg'] = df.groupby('country').alcohol.transform('bfill')
    df.alcohol = df.alcohol.fillna(df.country_avg)


    # fill null for total expenditure using backfill method
    df['country_avg'] = df.groupby('country').total_expenditure.transform('bfill')
    df.total_expenditure = df.total_expenditure.fillna(df.country_avg)


    # some countries has 0 entries of yrs of education, to save this column I filled NAs using the
    # mean for developed countries and undeveloped countries
    df['country_avg'] = df.groupby('developed_country').yrs_education.transform('mean')
    df.yrs_education = df.yrs_education.fillna(df.country_avg)


    # population and gdp column has too many to impute and the data is wrong, distribution is all over the place
    df.drop(columns=['population','gdp', 'income_comp_resources', 'country_avg'], inplace=True)


    # drop the remaining nulls in the columns
    df.dropna(inplace=True)

    return df


def value_counts(df, column):
    '''
    This function takes in a dataframe and list of columns and prints value counts for each column.
    '''
    for col in column:
        print(col)
        print(df[col].value_counts())
        print('-------------')


def show_outliers(df, k, columns):
    '''
    calculates the lower and upper bound to locate outliers and displays them
    note: recommended k be 1.5 and entered as integer
    '''
    for i in columns:
        quartile1, quartile3 = np.percentile(df[i], [25,75])
        IQR_value = quartile3 - quartile1
        lower_bound = (quartile1 - (k * IQR_value))
        upper_bound = (quartile3 + (k * IQR_value))
        print(f'For {i} the lower bound is {lower_bound} and  upper bound is {upper_bound}')


def remove_outliers(df, k, columns):
    '''
    calculates the lower and upper bound to locate outliers in variables and then removes them.
    note: recommended k be 1.5 and entered as integer
    '''
    for i in columns:
        quartile1, quartile3 = np.percentile(df[i], [25,75])
        IQR_value = quartile3 - quartile1
        lower_bound = (quartile1 - (k * IQR_value))
        upper_bound = (quartile3 + (k * IQR_value))
        print(f'For {i} the lower bound is {lower_bound} and  upper bound is {upper_bound}')
    
    
        df = df[(df[i] <= upper_bound) & (df[i] >= lower_bound)]
        print('-----------------')
        print('Dataframe now has ', df.shape[0], 'rows and ', df.shape[1], 'columns')
    return df


### IMPUTER Function
def impute(df, strategy_method, column_list):
    ''' take in a df, strategy, and cloumn list
        return df with listed columns imputed using input stratagy
    '''
        
    imputer = SimpleImputer(strategy=strategy_method)  # build imputer

    df[column_list] = imputer.fit_transform(df[column_list]) # fit/transform selected columns

    return df


def split_data(df):
    '''
    This function takes in a dataframe and splits it into train, test, and 
    validate dataframes for my model
    '''

    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123)

    print('train--->', train.shape)
    print('validate--->', validate.shape)
    print('test--->', test.shape)
    return train, validate, test




def split_ytarget(train, validate, test, y_target):
    '''
    This function takes in split data and a y_target variable and creates X and y
    dataframes for each. Note: enter y_target as a string string
    '''
    X_train, y_train = train.drop(columns=[y_target]), train[y_target]
    X_validate, y_validate = validate.drop(columns=[y_target]), validate[y_target]
    X_test, y_test = test.drop(columns=[y_target]), test[y_target]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test





## Robust SCALER FUNCTION
def robust_scaler(X_train, X_validate, X_test, numeric_cols):
    """
    this function takes in 3 dataframes with the same columns,
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a robust scaler to the first dataframe and transforms all
    3 dataframes using that scaler.
    it returns 3 dataframes with the same column names and scaled values.
    """
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).

    scaler = RobustScaler(copy=True).fit(X_train[numeric_cols])

    # scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train.
    #
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=numeric_cols).set_index(
        [X_train.index.values]
    )

    X_validate_scaled = pd.DataFrame(
        X_validate_scaled_array, columns=numeric_cols
    ).set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=numeric_cols).set_index(
        [X_test.index.values]
    )

    # Overwriting columns in our input dataframes for simplicity
    for i in numeric_cols:
        X_train[i] = X_train_scaled[i]
        X_validate[i] = X_validate_scaled[i]
        X_test[i] = X_test_scaled[i]

    return X_train, X_validate, X_test, scaler


