from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    """
    Priprema podataka za modelovanje - radi feature engineering.
    """
    
    # Kreiranje kopije podataka
    df_processed = data.copy()

    # Uklanjanje nepotrebnih kolona
    columns_to_drop = ['id', 'date']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df_processed.columns]
    if existing_columns_to_drop:
        df_processed = df_processed.drop(existing_columns_to_drop, axis=1)
        print(f"Uklonjene kolone: {existing_columns_to_drop}")
    
    new_features_count = 0
    
    # Starost kuće (samo ako yr_built postoji)
    if 'yr_built' in df_processed.columns:
        df_processed['house_age'] = 2025 - df_processed['yr_built']
        new_features_count += 1
    
    # Da li je kuća renovirana (samo ako yr_renovated postoji)
    if 'yr_renovated' in df_processed.columns:
        df_processed['is_renovated'] = (df_processed['yr_renovated'] > 0).astype(int)
        new_features_count += 1
    
    # # Cena po kvadratnom futu
    # df_processed['price_per_sqft'] = df_processed['price'] / df_processed['sqft_living']
    # new_features_count += 1
    
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
    
    # Ispiši samo nove feature-e (one koje smo dodali u odnosu na originalne kolone)
    original_columns = list(data.columns)
    new_features = [col for col in df_processed.columns if col not in original_columns]
    print(f"Kreirano {len(new_features)} novih feature-a: {new_features}")
    
    # Uklanjanje 'price_per_sqft' jer je korišćen target za njegovo kreiranje
    # if 'price_per_sqft' in df_processed.columns:
    #     df_processed = df_processed.drop('price_per_sqft', axis=1)
    
    return df_processed

def split_data(X, y, test_size=0.15, random_state=42):
    """
    Deli podatke na trening i test skup.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    print(f"\nPODELA PODATAKA:")
    print(f"Train set: {X_train.shape[0]} uzoraka ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Test set: {X_test.shape[0]} uzoraka ({X_test.shape[0]/len(X)*100:.1f}%)")
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    """
    Skalira podatke koristeći StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("\nSkalirani su podaci")
    return X_train_scaled, X_test_scaled, scaler