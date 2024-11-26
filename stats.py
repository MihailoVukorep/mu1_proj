#!/usr/bin/env python3

# %% init

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

df['bedrooms'].unique()

# %%
df[df['bedrooms'] == 33]

# %%
a = df['waterfront'].unique()

# %%

plt.hist(df['price'], bins=100, density=True)


# %%
sb.pairplot(df)
# %%
