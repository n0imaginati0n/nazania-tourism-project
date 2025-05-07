
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer

def map_travel_with(row: pd.Series, not_alone_value: str) -> str:
    if pd.isnull(row['travel_with']):
        if row['total_male'] + row['total_female'] == 1:
            return 'Alone'
        else:
            return not_alone_value
    else:
        return row['travel_with']

def get_preprocessed_data(file_name: str, not_alone_value = 'Spouse'):
    """ load and preprocess data

    Args:
        not_alone_value (str, optional): the value, which should be used for 
            non-alone travellers in 'travel_with' column. Defaults to 'Spouse'.

    Returns:
        _type_: DataFrame with preprocessed data
    """    
    df = pd.read_csv(file_name)
    df = df.drop(['ID'], axis = 1)
    df['travel_with'] = df.apply(lambda row: map_travel_with(row, not_alone_value), axis = 1)
    df['most_impressing'] = df['most_impressing'].fillna('No comments')
    df = df.dropna(how="any", axis = 0)
    df = pd.get_dummies(df, drop_first=True)
    return df

