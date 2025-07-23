# ===================================================================
# MODUL ZA UČITAVANJE I EKSPLORACIJU PODATAKA
# ===================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(csv_path):
    """
    Učitava podatke iz CSV fajla
    """
    try:
        data = pd.read_csv(csv_path)
        print(f"✅ Uspešno učitani podaci iz {csv_path}")
        print(f"📊 Oblik dataseta: {data.shape}")
        return data
    except Exception as e:
        print(f"❌ Greška pri učitavanju: {e}")
        return None

def explore_data(data):
    """
    Eksplorativna analiza podataka
    """
    print("\n" + "="*50)
    print("📈 EKSPLORATIVNA ANALIZA PODATAKA")
    print("="*50)
    
    print(f"📏 Oblik dataseta: {data.shape}")
    print(f"\n📋 Tipovi podataka:")
    print(data.dtypes)
    
    print(f"\n📊 Osnovne statistike za cenu:")
    print(data['price'].describe())
    
    print(f"\n❌ Nedostajuće vrednosti:")
    missing_values = data.isnull().sum()
    print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "Nema nedostajućih vrednosti!")
    
    # Kreiranje figure sa subplotovima
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Distribucija cene
    plt.subplot(3, 3, 1)
    plt.hist(data['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribucija cena kuća', fontsize=14, fontweight='bold')
    plt.xlabel('Cena ($)')
    plt.ylabel('Frekvencija')
    plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    # 2. Log distribucija cene
    plt.subplot(3, 3, 2)
    plt.hist(np.log10(data['price']), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Log distribucija cena', fontsize=14, fontweight='bold')
    plt.xlabel('Log10(Cena)')
    plt.ylabel('Frekvencija')
    
    # 3. Korelaciona matrica (top 10 korelacija sa cenom)
    plt.subplot(3, 3, 3)
    corr_matrix = data.select_dtypes(include=[np.number]).corr()
    price_corr = corr_matrix['price'].abs().sort_values(ascending=False).head(10)
    price_corr_vals = corr_matrix['price'][price_corr.index]
    
    colors = ['red' if x < 0 else 'green' for x in price_corr_vals]
    plt.barh(range(len(price_corr_vals)), price_corr_vals.values, color=colors, alpha=0.7)
    plt.yticks(range(len(price_corr_vals)), price_corr_vals.index, fontsize=10)
    plt.xlabel('Korelacija sa cenom')
    plt.title('Top 10 korelacija sa cenom', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # 4. Kvadratura vs Cena
    plt.subplot(3, 3, 4)
    plt.scatter(data['sqft_living'], data['price'], alpha=0.5, s=1)
    plt.xlabel('Kvadratura stambenog prostora (sqft)')
    plt.ylabel('Cena ($)')
    plt.title('Kvadratura vs Cena', fontsize=14, fontweight='bold')
    
    # 5. Broj spavaćih soba vs Prosečna cena
    plt.subplot(3, 3, 5)
    bedroom_price = data.groupby('bedrooms')['price'].mean()
    plt.bar(bedroom_price.index, bedroom_price.values, color='orange', alpha=0.7)
    plt.xlabel('Broj spavaćih sobe')
    plt.ylabel('Prosečna cena ($)')
    plt.title('Spavaće sobe vs Prosečna cena', fontsize=14, fontweight='bold')
    
    # 6. Grade vs Prosečna cena
    plt.subplot(3, 3, 6)
    grade_price = data.groupby('grade')['price'].mean()
    plt.bar(grade_price.index, grade_price.values, color='purple', alpha=0.7)
    plt.xlabel('Grade (kvalitet)')
    plt.ylabel('Prosečna cena ($)')
    plt.title('Grade vs Prosečna cena', fontsize=14, fontweight='bold')
    
    # 7. Waterfront efekat
    plt.subplot(3, 3, 7)
    waterfront_price = data.groupby('waterfront')['price'].mean()
    labels = ['Bez pristupa vodi', 'Sa pristupom vodi']
    plt.bar(labels, waterfront_price.values, color=['lightblue', 'darkblue'], alpha=0.7)
    plt.ylabel('Prosečna cena ($)')
    plt.title('Efekat pristupa vodi', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    
    # 8. Godina izgradnje vs Cena
    plt.subplot(3, 3, 8)
    plt.scatter(data['yr_built'], data['price'], alpha=0.5, s=1, color='red')
    plt.xlabel('Godina izgradnje')
    plt.ylabel('Cena ($)')
    plt.title('Godina izgradnje vs Cena', fontsize=14, fontweight='bold')
    
    # 9. Geografska distribucija (lat vs long, color = price)
    plt.subplot(3, 3, 9)
    scatter = plt.scatter(data['long'], data['lat'], 
                        c=data['price'], cmap='viridis', alpha=0.6, s=1)
    plt.xlabel('Geografska dužina')
    plt.ylabel('Geografska širina')
    plt.title('Geografska distribucija cena', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Cena ($)')
    
    plt.tight_layout()
    plt.show()
    
    # Dodatne statistike
    print(f"\n💰 CENOVNE STATISTIKE:")
    print(f"Najjeftinija kuća: ${data['price'].min():,.2f}")
    print(f"Najskuplja kuća: ${data['price'].max():,.2f}")
    print(f"Prosečna cena: ${data['price'].mean():,.2f}")
    print(f"Medijana cene: ${data['price'].median():,.2f}")
    
    print(f"\n🏠 KARAKTERISTIKE KUĆA:")
    print(f"Prosečna kvadratura: {data['sqft_living'].mean():.0f} sqft")
    print(f"Prosečan broj spavaćih soba: {data['bedrooms'].mean():.1f}")
    print(f"Prosečan broj kupatila: {data['bathrooms'].mean():.1f}")
    print(f"Kuće sa pristupom vodi: {(data['waterfront'].sum() / len(data) * 100):.1f}%")

def create_correlation_plot(data):
    """
    Kreira plot korelacije sa cenom
    """
    numeric_data = data.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    price_corr = corr_matrix['price'].drop('price').sort_values(key=np.abs, ascending=False)
    
    plt.figure(figsize=(8, 8))
    sns.barplot(x=price_corr.values, y=price_corr.index, palette='coolwarm')
    plt.title('Korelacija feature-a sa cenom (price)', fontsize=16)
    plt.xlabel('Korelacija sa cenom')
    plt.ylabel('Feature')
    plt.grid(axis='x', alpha=0.3)
    plt.show()
    
    return price_corr

def select_features_by_correlation(data, correlation_limit=0.15):
    """
    Selektuje feature-e na osnovu korelacije sa cenom
    """
    numeric_data = data.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    price_corr = corr_matrix['price'].drop('price').sort_values(key=np.abs, ascending=False)
    
    selected_features = price_corr[price_corr.abs() >= correlation_limit].index.tolist()
    print(f"Feature-i sa |korelacijom| >= {correlation_limit}: {selected_features}")

    dropped_features = [col for col in numeric_data.columns if col not in selected_features + ['price']]
    print(f"Izbačene kolone: {dropped_features}")

    selected_df = data[selected_features + ['price']]
    print(f"Oblik novog dataframe-a za učenje: {selected_df.shape}")
    
    return selected_df, selected_features, dropped_features
