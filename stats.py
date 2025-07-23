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

# 1. Provera negativnih vrednosti za numeričke kolone
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
invalid_negative_values = df[numeric_columns].lt(0).any(axis=1)

# 2. Provera NaN (null) vrednosti
invalid_null_values = df.isnull().any(axis=1)

# 3. Provera vrednosti koje nisu numeričke u numeričkim kolonama
invalid_non_numeric_values = pd.DataFrame()
for col in numeric_columns:
    invalid_non_numeric_values[col] = pd.to_numeric(df[col], errors='coerce').isnull()

# 4. Provera vrednosti van prihvatljivih opsega (primer: cena manja od 0 i godina izgradnje u budućnosti)
invalid_price_range = df['price'] < 0
invalid_year_built_range = df['yr_built'] > 2024  # Pretpostavljamo da je trenutna godina 2024

# 5. Provera duplikata
duplicate_rows = df.duplicated()

# 6. Provera nekonzistentnih vrednosti (npr. format broja ili datuma)
# Ovaj deo zavisi od specifičnog formata podataka i može biti specifičan za vaš dataset

# Prikazivanje rezultata
print("Nevalidne vrednosti (negativne vrednosti):")
print(df[invalid_negative_values])

print("\nNevalidne vrednosti (NaN vrednosti):")
print(df[invalid_null_values])

print("\nNevalidne vrednosti (ne-numeričke vrednosti u numeričkim kolonama):")
print(df[invalid_non_numeric_values.any(axis=1)])

print("\nNevalidne vrednosti (cena manja od 0):")
print(df[invalid_price_range])

print("\nNevalidne vrednosti (godina izgradnje u budućnosti):")
print(df[invalid_year_built_range])

print("\nDuplikati u podacima:")
print(df[duplicate_rows])
# %%
