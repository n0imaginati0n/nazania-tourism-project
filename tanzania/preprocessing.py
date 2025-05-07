
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

def get_preprocessed_data(file_name: str, not_alone_value: str = 'Spouse') -> pd.DataFrame:
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
    return pd.get_dummies(df, drop_first=True)

if __name__ == '__main__':
    df = get_preprocessed_data('data/Train.csv')
    print(f'loaded {df.shape[1]} columns, {df.shape[0]} rows')
