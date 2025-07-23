# ===================================================================
# MODUL ZA PRIPREMU PODATAKA
# ===================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    """
    Priprema podataka za modelovanje
    """
    print("\n" + "="*50)
    print("🔧 PRIPREMA PODATAKA")
    print("="*50)
    
    # Kreiranje kopije podataka
    df_processed = data.copy()
    
    # Kreiranje novih feature-a (feature engineering)
    print("🛠️ Kreiranje novih feature-a...")
    
    new_features_count = 0
    
    # Starost kuće (samo ako yr_built postoji)
    if 'yr_built' in df_processed.columns:
        df_processed['house_age'] = 2025 - df_processed['yr_built']
        new_features_count += 1
    
    # Da li je kuća renovirana (samo ako yr_renovated postoji)
    if 'yr_renovated' in df_processed.columns:
        df_processed['is_renovated'] = (df_processed['yr_renovated'] > 0).astype(int)
        new_features_count += 1
    
    # Cena po kvadratnom futu
    df_processed['price_per_sqft'] = df_processed['price'] / df_processed['sqft_living']
    new_features_count += 1
    
    # Procenat podruma u odnosu na ukupnu kvadraturu
    df_processed['basement_ratio'] = df_processed['sqft_basement'] / df_processed['sqft_living']
    new_features_count += 1
    
    # Da li je kuća velika (preko proseka)
    avg_sqft = df_processed['sqft_living'].mean()
    df_processed['is_large_house'] = (df_processed['sqft_living'] > avg_sqft).astype(int)
    new_features_count += 1
    
    # Kombinovani feature za luksuz (waterfront + high grade + view)
    df_processed['luxury_score'] = (
        df_processed['waterfront'] * 3 + 
        (df_processed['grade'] >= 10).astype(int) * 2 + 
        (df_processed['view'] >= 3).astype(int)
    )
    new_features_count += 1
    
    print(f"✅ Kreirano {new_features_count} novih feature-a")
    print(f"📊 Ukupan broj feature-a: {df_processed.shape[1] - 1}")  # -1 jer ne računamo target
    
    # Podela na features i target
    X = df_processed.drop(['price', 'price_per_sqft'], axis=1)  # price_per_sqft ne koristimo jer koristi target
    y = df_processed['price']
    
    print(f"🎯 Target varijabla: price")
    print(f"📋 Feature-i: {list(X.columns)}")
    
    # Podela na train/test (85%/15%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=None
    )
    
    print(f"\n📊 PODELA PODATAKA:")
    print(f"Train set: {X_train.shape[0]} uzoraka ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Test set: {X_test.shape[0]} uzoraka ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # Skaliranje podataka
    print("⚖️ Skaliranje podataka...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Podaci su uspešno pripremljeni za modelovanje!")
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler
