import pandas as pd
import numpy as np

def one_hot_code_strings(df):
    count = 0
    for i in df:
        if type(df[i][0]) is str:        
            dummies = pd.get_dummies(df[i])
            for j in dummies:
                name = 'one_hot_coded_' + str(count)
                df[name] = dummies[j]
                count += 1
    t = df
    for i in t:
        if type(t[i][0]) is str:
            t = t.drop(i, 1)
    df = t
    return df