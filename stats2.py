# %%
import pandas as pd

# Učitavanje skupa podataka
df = pd.read_csv("dataset/kc_house_data.csv")

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
