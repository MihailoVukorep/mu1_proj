#!/usr/bin/env python3

# %% init

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# %%

df = pd.read_csv("dataset/kc_house_data.csv")

# %%

df.head()

# %%

df.shape

# %%

for feature in df.columns:
    print(f"{feature}: {df[feature].nunique()}")

# %%

for feature in df.columns:
    if df[feature].nunique() < 30:
        print(f"{feature}: {df[feature].unique()}")

# %%

plt.hist(df['price'], bins=100, density=True)


# %%
#pairplot = sb.pairplot(df)
#pairplot.savefig("pairplot.png")
# %%

df.dtypes

# %%

# count of numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(len(numeric_cols))  
# %%


def check_column_type(df):
    for column in df.columns:
        if df[column].dtype == 'object' or pd.api.types.is_categorical_dtype(df[column]):
            print(f"{column} je kategorijsko obeležje.")
        elif pd.api.types.is_numeric_dtype(df[column]):
            print(f"{column} je numeričko obeležje.")
check_column_type(df)

# %%

print(df['date'].value_counts())

# %%

print(str(df['date'].unique()))
# %%

pd.options.display.float_format = '{:,.2f}'.format
print(df['price'].describe())
# %%

# %%

print(df['id'].value_counts())
# %%


print(df[df['id'] == 795000620])
# %%


df.isna().sum()

# %%

df.describe()


# %%

df[df['bedrooms'] == 0]

# %%


print(df['view'].value_counts())
# %%
