# from https://github.com/ohmeow/pandas_examples

# import sys
# sys.path.append('/home/stas/fast.ai')
# from myutils.pandas_util import advanced_describe

import pandas as pd


######################### Data Examination ############################ 
# - made changes to unique_vals to show a small sample, regardless how many there are
def advanced_describe(df):
    # get descriptive stats for dataframe for 'all' column dtypes
    desc = df.describe(include='all').T
    desc.drop(['top', 'freq', 'unique'], axis=1, inplace=True)
    
    # update column counts (df.describe() returns NaN for non-numeric cols)
    counts = pd.Series({ col: df[col].count() for col in df.columns })
    desc.update(counts.to_frame('count'))
    
    # add missing count/%
    missings = df.isnull().sum()
    desc = pd.concat([desc, missings.to_frame('missing')], axis=1)
    desc['missing%'] = (desc['missing'] / len(desc)).round(2)

    # add unique counts/%
    uniques = pd.Series({ col: len(df[col].unique()) for col in df.columns })
    desc = pd.concat([desc, uniques.to_frame('unique')], axis=1)
    desc['unique%'] = (desc['unique'] / len(desc)).round(2)
    
    unique_vals = pd.Series({ col: df[col].unique() if len(df[col].unique()) < 10 else [*df[col].unique()[0:10],"..."] for col in df.columns })
    desc = pd.concat([desc, unique_vals.to_frame('unique_values')], axis=1, sort=True)
    
    # add col dtype
    dtypes = pd.Series({ col: df[col].dtype for col in df.columns })
    desc = pd.concat([desc, dtypes.to_frame('dtype')], axis=1, sort=True)
    
    return desc

# same as advanced_describe but with fever attributes to avoid
# horizontal scrolling
def advanced_describe_short(df):
    # get descriptive stats for dataframe for 'all' column dtypes
    desc = df.describe(include='all').T
    desc.drop(['top', 'freq', 'unique', '25%', '50%', '75%'], axis=1, inplace=True)
    
    # update column counts (df.describe() returns NaN for non-numeric cols)
    counts = pd.Series({ col: df[col].count() for col in df.columns })
    desc.update(counts.to_frame('count'))
    
    # add missing count/%
    missings = df.isnull().sum()
    desc = pd.concat([desc, missings.to_frame('missing')], axis=1)
    #desc['missing%'] = (desc['missing'] / len(desc)).round(2)

    # add unique counts/%
    uniques = pd.Series({ col: len(df[col].unique()) for col in df.columns })
    desc = pd.concat([desc, uniques.to_frame('unique')], axis=1)
    #desc['unique%'] = (desc['unique'] / len(desc)).round(2)
    
    unique_vals = pd.Series({ col: df[col].unique() if len(df[col].unique()) < 10 else [*df[col].unique()[0:10],"..."] for col in df.columns })
    desc = pd.concat([desc, unique_vals.to_frame('unique_values')], axis=1, sort=True)
    
    # add col dtype
    dtypes = pd.Series({ col: df[col].dtype for col in df.columns })
    desc = pd.concat([desc, dtypes.to_frame('dtype')], axis=1, sort=True)
    
    return desc


######################### Data Cleaning and Preparation ###############

def fillna_by_group(df, target_col, group_cols, agg='median'):
    df[target_col] = df.groupby(group_cols)[target_col].transform(lambda x: x.fillna(eval(f'x.{agg}()')))


######################### Feature Engineering ######################### 

def add_by_regex(df, target_col, new_col, regex):
        df[new_col] = df[target_col].str.extract(regex, expand=False)
