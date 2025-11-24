#utility functions for preprocessing dataset

import pandas as pd

#clean_data(): clean input dataframe by handling missing values and duplicates
def clean_data(df):
    
    #drop duplicate rows
    df = df.drop_duplicates()

    #fill numeric missing values with median
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    #fill categorical missing values with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0])

    return df

#encode_categoricals(): convert categorical columns into numeric columns
def encode_categoricals(df):
    cat_cols = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df
