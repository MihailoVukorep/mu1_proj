# Projekat: Predviđanje cena nekretnina
# Autor: Mihailo Vukorep IN 40/2021, Kolarski Marko IN 60/2021

# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Postavljanje stila za grafikone
plt.style.use('default')
sns.set_palette("husl")

class HousePricePrediction:
    def __init__(self, csv_path):
        """
        Inicijalizacija klase za predviđanje cena nekretnina
        """
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.X_train_pca = None
        self.X_test_pca = None
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.models = {}
        self.results = {}
        
        self.load_data(csv_path)
    
    def load_data(self, csv_path):
        """
        Učitava podatke iz CSV fajla
        """
        try:
            self.data = pd.read_csv(csv_path)
            print(f"✅ Uspešno učitani podaci iz {csv_path}")
            print(f"📊 Oblik dataseta: {self.data.shape}")
        except Exception as e:
            print(f"❌ Greška pri učitavanju: {e}")
            return None
    
    def explore_data(self):
        """
        Eksplorativna analiza podataka
        """
        print("\n" + "="*50)
        print("📈 EKSPLORATIVNA ANALIZA PODATAKA")
        print("="*50)
        
        print(f"📏 Oblik dataseta: {self.data.shape}")
        print(f"\n📋 Tipovi podataka:")
        print(self.data.dtypes)
        
        print(f"\n📊 Osnovne statistike za cenu:")
        print(self.data['price'].describe())
        
        print(f"\n❌ Nedostajuće vrednosti:")
        missing_values = self.data.isnull().sum()
        print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "Nema nedostajućih vrednosti!")
        
        # Kreiranje figure sa subplotovima
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Distribucija cene
        plt.subplot(3, 3, 1)
        plt.hist(self.data['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribucija cena kuća', fontsize=14, fontweight='bold')
        plt.xlabel('Cena ($)')
        plt.ylabel('Frekvencija')
        plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        
        # 2. Log distribucija cene
        plt.subplot(3, 3, 2)
        plt.hist(np.log10(self.data['price']), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.title('Log distribucija cena', fontsize=14, fontweight='bold')
        plt.xlabel('Log10(Cena)')
        plt.ylabel('Frekvencija')
        
        # 3. Korelaciona matrica (top 10 korelacija sa cenom)
        plt.subplot(3, 3, 3)
        corr_matrix = self.data.select_dtypes(include=[np.number]).corr()
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
        plt.scatter(self.data['sqft_living'], self.data['price'], alpha=0.5, s=1)
        plt.xlabel('Kvadratura stambenog prostora (sqft)')
        plt.ylabel('Cena ($)')
        plt.title('Kvadratura vs Cena', fontsize=14, fontweight='bold')
        
        # 5. Broj spavaćih soba vs Prosečna cena
        plt.subplot(3, 3, 5)
        bedroom_price = self.data.groupby('bedrooms')['price'].mean()
        plt.bar(bedroom_price.index, bedroom_price.values, color='orange', alpha=0.7)
        plt.xlabel('Broj spavaćih soba')
        plt.ylabel('Prosečna cena ($)')
        plt.title('Spavaće sobe vs Prosečna cena', fontsize=14, fontweight='bold')
        
        # 6. Grade vs Prosečna cena
        plt.subplot(3, 3, 6)
        grade_price = self.data.groupby('grade')['price'].mean()
        plt.bar(grade_price.index, grade_price.values, color='purple', alpha=0.7)
        plt.xlabel('Grade (kvalitet)')
        plt.ylabel('Prosečna cena ($)')
        plt.title('Grade vs Prosečna cena', fontsize=14, fontweight='bold')
        
        # 7. Waterfront efekat
        plt.subplot(3, 3, 7)
        waterfront_price = self.data.groupby('waterfront')['price'].mean()
        labels = ['Bez pristupa vodi', 'Sa pristupom vodi']
        plt.bar(labels, waterfront_price.values, color=['lightblue', 'darkblue'], alpha=0.7)
        plt.ylabel('Prosečna cena ($)')
        plt.title('Efekat pristupa vodi', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        
        # 8. Godina izgradnje vs Cena
        plt.subplot(3, 3, 8)
        plt.scatter(self.data['yr_built'], self.data['price'], alpha=0.5, s=1, color='red')
        plt.xlabel('Godina izgradnje')
        plt.ylabel('Cena ($)')
        plt.title('Godina izgradnje vs Cena', fontsize=14, fontweight='bold')
        
        # 9. Geografska distribucija (lat vs long, color = price)
        plt.subplot(3, 3, 9)
        scatter = plt.scatter(self.data['long'], self.data['lat'], 
                            c=self.data['price'], cmap='viridis', alpha=0.6, s=1)
        plt.xlabel('Geografska dužina')
        plt.ylabel('Geografska širina')
        plt.title('Geografska distribucija cena', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Cena ($)')
        
        plt.tight_layout()
        plt.show()
        
        # Dodatne statistike
        print(f"\n💰 CENOVNE STATISTIKE:")
        print(f"Najjeftinija kuća: ${self.data['price'].min():,.2f}")
        print(f"Najskuplja kuća: ${self.data['price'].max():,.2f}")
        print(f"Prosečna cena: ${self.data['price'].mean():,.2f}")
        print(f"Medijana cene: ${self.data['price'].median():,.2f}")
        
        print(f"\n🏠 KARAKTERISTIKE KUĆA:")
        print(f"Prosečna kvadratura: {self.data['sqft_living'].mean():.0f} sqft")
        print(f"Prosečan broj spavaćih soba: {self.data['bedrooms'].mean():.1f}")
        print(f"Prosečan broj kupatila: {self.data['bathrooms'].mean():.1f}")
        print(f"Kuće sa pristupom vodi: {(self.data['waterfront'].sum() / len(self.data) * 100):.1f}%")
    
    def preprocess_data(self):
        """
        Priprema podataka za modelovanje
        """
        print("\n" + "="*50)
        print("🔧 PRIPREMA PODATAKA")
        print("="*50)
        
        # Kreiranje kopije podataka
        df_processed = self.data.copy()
        
        # Uklanjanje nepotrebnih kolona
        columns_to_drop = ['id', 'date']
        df_processed = df_processed.drop(columns_to_drop, axis=1)
        print(f"🗑️ Uklonjene kolone: {columns_to_drop}")
        
        # Kreiranje novih feature-a (feature engineering)
        print("🛠️ Kreiranje novih feature-a...")
        
        # Starost kuće
        df_processed['house_age'] = 2024 - df_processed['yr_built']
        
        # Da li je kuća renovirana
        df_processed['is_renovated'] = (df_processed['yr_renovated'] > 0).astype(int)
        
        # Cena po kvadratnom futu
        df_processed['price_per_sqft'] = df_processed['price'] / df_processed['sqft_living']
        
        # Procenat podruma u odnosu na ukupnu kvadraturu
        df_processed['basement_ratio'] = df_processed['sqft_basement'] / df_processed['sqft_living']
        
        # Da li je kuća velika (preko proseka)
        avg_sqft = df_processed['sqft_living'].mean()
        df_processed['is_large_house'] = (df_processed['sqft_living'] > avg_sqft).astype(int)
        
        # Kombinovani feature za luksuz (waterfront + high grade + view)
        df_processed['luxury_score'] = (
            df_processed['waterfront'] * 3 + 
            (df_processed['grade'] >= 10).astype(int) * 2 + 
            (df_processed['view'] >= 3).astype(int)
        )
        
        print(f"✅ Kreirano 6 novih feature-a")
        print(f"📊 Ukupan broj feature-a: {df_processed.shape[1] - 1}")  # -1 jer ne računamo target
        
        # Podela na features i target
        X = df_processed.drop(['price', 'price_per_sqft'], axis=1)  # price_per_sqft ne koristimo jer koristi target
        y = df_processed['price']
        
        print(f"🎯 Target varijabla: price")
        print(f"📋 Feature-i: {list(X.columns)}")
        
        # Podela na train/test (85%/15%)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=None
        )
        
        print(f"\n📊 PODELA PODATAKA:")
        print(f"Train set: {self.X_train.shape[0]} uzoraka ({self.X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"Test set: {self.X_test.shape[0]} uzoraka ({self.X_test.shape[0]/len(X)*100:.1f}%)")
        
        # Skaliranje podataka
        print("⚖️ Skaliranje podataka...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("✅ Podaci su uspešno pripremljeni za modelovanje!")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def apply_pca(self, n_components=0.95):
        """
        Primenjuje PCA za smanjenje dimenzionalnosti
        """
        print("\n" + "="*50)
        print("🔍 PRIMENA PCA")
        print("="*50)
        
        self.pca = PCA(n_components=n_components, random_state=42)
        self.X_train_pca = self.pca.fit_transform(self.X_train_scaled)
        self.X_test_pca = self.pca.transform(self.X_test_scaled)
        
        print(f"📊 Originalan broj feature-a: {self.X_train_scaled.shape[1]}")
        print(f"🎯 Broj komponenti nakon PCA: {self.X_train_pca.shape[1]}")
        print(f"📈 Objašnjena varijansa: {self.pca.explained_variance_ratio_.sum():.3f} ({self.pca.explained_variance_ratio_.sum()*100:.1f}%)")
        
        # Vizuelizacija objašnjene varijanse
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(self.pca.explained_variance_ratio_) + 1), 
                self.pca.explained_variance_ratio_, alpha=0.7, color='skyblue')
        plt.xlabel('Komponenta')
        plt.ylabel('Objašnjena varijansa')
        plt.title('Objašnjena varijansa po komponenti')
        plt.grid(axis='y', alpha=0.3)
        
        plt.subplot(1, 2, 2)
        cumsum = np.cumsum(self.pca.explained_variance_ratio_)
        plt.plot(range(1, len(cumsum) + 1), cumsum, 'bo-', linewidth=2, markersize=6)
        plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% varijanse')
        plt.xlabel('Broj komponenti')
        plt.ylabel('Kumulativna objašnjena varijansa')
        plt.title('Kumulativna objašnjena varijansa')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Prikaz najvažnijih feature-a za prve komponente
        feature_names = self.X_train.columns
        print(f"\n🏆 NAJVAŽNIJI FEATURE-I ZA PRVE 3 KOMPONENTE:")
        
        for i in range(min(3, self.pca.n_components_)):
            component = self.pca.components_[i]
            top_features_idx = np.argsort(np.abs(component))[-5:][::-1]
            
            print(f"\nKomponenta {i+1}:")
            for j, idx in enumerate(top_features_idx):
                print(f"  {j+1}. {feature_names[idx]}: {component[idx]:.3f}")
        
        return self.X_train_pca, self.X_test_pca
    
    def train_models(self):
        """
        Trenira različite modele sa cross-validation i hyperparameter tuning
        """
        print("\n" + "="*50)
        print("🤖 TRENIRANJE MODELA")
        print("="*50)
        
        # Definisanje modela
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'SVR': SVR(),
            'KNN': KNeighborsRegressor()
        }
        
        # Hiperparametri za Grid Search
        param_grids = {
            'Linear Regression': {},
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'SVR': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'epsilon': [0.01, 0.1]
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 10],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
        
        self.results = {}
        
        for name, model in models.items():
            print(f"\n🔄 Treniranje {name}...")
            
            # Grid Search sa Cross Validation za originalne podatke
            if param_grids[name]:
                print("  🔍 Pretraga hiperparametara (originalni prostor)...")
                grid_search_orig = GridSearchCV(
                    model, param_grids[name], 
                    cv=5, scoring='neg_mean_squared_error', 
                    n_jobs=-1, verbose=0
                )
                grid_search_orig.fit(self.X_train_scaled, self.y_train)
                best_model_orig = grid_search_orig.best_estimator_
                best_params_orig = grid_search_orig.best_params_
                
                print("  🔍 Pretraga hiperparametara (PCA prostor)...")
                grid_search_pca = GridSearchCV(
                    model, param_grids[name], 
                    cv=5, scoring='neg_mean_squared_error', 
                    n_jobs=-1, verbose=0
                )
                grid_search_pca.fit(self.X_train_pca, self.y_train)
                best_model_pca = grid_search_pca.best_estimator_
                best_params_pca = grid_search_pca.best_params_
                
            else:
                # Za Linear Regression nema hiperparametara
                best_model_orig = model
                best_model_orig.fit(self.X_train_scaled, self.y_train)
                best_params_orig = {}
                
                best_model_pca = LinearRegression()
                best_model_pca.fit(self.X_train_pca, self.y_train)
                best_params_pca = {}
            
            # Cross-validation rezultati
            print("  📊 Cross-validation evaluacija...")
            cv_scores_orig = cross_val_score(
                best_model_orig, self.X_train_scaled, self.y_train, 
                cv=5, scoring='neg_mean_squared_error'
            )
            cv_scores_pca = cross_val_score(
                best_model_pca, self.X_train_pca, self.y_train, 
                cv=5, scoring='neg_mean_squared_error'
            )
            
            # R2 score cross-validation
            cv_r2_orig = cross_val_score(
                best_model_orig, self.X_train_scaled, self.y_train, 
                cv=5, scoring='r2'
            )
            cv_r2_pca = cross_val_score(
                best_model_pca, self.X_train_pca, self.y_train, 
                cv=5, scoring='r2'
            )
            
            self.results[name] = {
                'model_original': best_model_orig,
                'model_pca': best_model_pca,
                'best_params_original': best_params_orig,
                'best_params_pca': best_params_pca,
                'cv_mse_original': -cv_scores_orig.mean(),
                'cv_mse_pca': -cv_scores_pca.mean(),
                'cv_mse_std_original': cv_scores_orig.std(),
                'cv_mse_std_pca': cv_scores_pca.std(),
                'cv_r2_original': cv_r2_orig.mean(),
                'cv_r2_pca': cv_r2_pca.mean(),
                'cv_r2_std_original': cv_r2_orig.std(),
                'cv_r2_std_pca': cv_r2_pca.std()
            }
            
            print(f"  ✅ {name} - Završeno!")
            print(f"     📈 CV MSE (Original): {self.results[name]['cv_mse_original']:,.0f} ± {self.results[name]['cv_mse_std_original']:,.0f}")
            print(f"     📈 CV MSE (PCA): {self.results[name]['cv_mse_pca']:,.0f} ± {self.results[name]['cv_mse_std_pca']:,.0f}")
            print(f"     📊 CV R² (Original): {self.results[name]['cv_r2_original']:.3f} ± {self.results[name]['cv_r2_std_original']:.3f}")
            print(f"     📊 CV R² (PCA): {self.results[name]['cv_r2_pca']:.3f} ± {self.results[name]['cv_r2_std_pca']:.3f}")
        
        # Visualizacija cross-validation rezultata
        self.plot_cv_results()
        
        return self.results
    
    def plot_cv_results(self):
        """
        Vizuelizuje cross-validation rezultate
        """
        model_names = list(self.results.keys())
        cv_mse_orig = [self.results[name]['cv_mse_original'] for name in model_names]
        cv_mse_pca = [self.results[name]['cv_mse_pca'] for name in model_names]
        cv_r2_orig = [self.results[name]['cv_r2_original'] for name in model_names]
        cv_r2_pca = [self.results[name]['cv_r2_pca'] for name in model_names]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # MSE poređenje
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0,0].bar(x - width/2, cv_mse_orig, width, label='Original', alpha=0.8, color='skyblue')
        axes[0,0].bar(x + width/2, cv_mse_pca, width, label='PCA', alpha=0.8, color='lightcoral')
        axes[0,0].set_xlabel('Modeli')
        axes[0,0].set_ylabel('Cross-Validation MSE')
        axes[0,0].set_title('MSE poređenje - Original vs PCA')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(model_names, rotation=45)
        axes[0,0].legend()
        axes[0,0].grid(axis='y', alpha=0.3)
        
        # R2 poređenje
        axes[0,1].bar(x - width/2, cv_r2_orig, width, label='Original', alpha=0.8, color='lightgreen')
        axes[0,1].bar(x + width/2, cv_r2_pca, width, label='PCA', alpha=0.8, color='orange')
        axes[0,1].set_xlabel('Modeli')
        axes[0,1].set_ylabel('Cross-Validation R²')
        axes[0,1].set_title('R² poređenje - Original vs PCA')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(model_names, rotation=45)
        axes[0,1].legend()
        axes[0,1].grid(axis='y', alpha=0.3)
        
        # MSE sa error bars
        mse_orig_std = [self.results[name]['cv_mse_std_original'] for name in model_names]
        mse_pca_std = [self.results[name]['cv_mse_std_pca'] for name in model_names]
        
        axes[1,0].errorbar(range(len(model_names)), cv_mse_orig, yerr=mse_orig_std, 
                          marker='o', capsize=5, label='Original', linewidth=2, markersize=8)
        axes[1,0].errorbar(range(len(model_names)), cv_mse_pca, yerr=mse_pca_std, 
                          marker='s', capsize=5, label='PCA', linewidth=2, markersize=8)
        axes[1,0].set_xlabel('Modeli')
        axes[1,0].set_ylabel('Cross-Validation MSE')
        axes[1,0].set_title('MSE sa standardnom devijacijom')
        axes[1,0].set_xticks(range(len(model_names)))
        axes[1,0].set_xticklabels(model_names, rotation=45)
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # R2 sa error bars
        r2_orig_std = [self.results[name]['cv_r2_std_original'] for name in model_names]
        r2_pca_std = [self.results[name]['cv_r2_std_pca'] for name in model_names]
        
        axes[1,1].errorbar(range(len(model_names)), cv_r2_orig, yerr=r2_orig_std, 
                          marker='o', capsize=5, label='Original', linewidth=2, markersize=8)
        axes[1,1].errorbar(range(len(model_names)), cv_r2_pca, yerr=r2_pca_std, 
                          marker='s', capsize=5, label='PCA', linewidth=2, markersize=8)
        axes[1,1].set_xlabel('Modeli')
        axes[1,1].set_ylabel('Cross-Validation R²')
        axes[1,1].set_title('R² sa standardnom devijacijom')
        axes[1,1].set_xticks(range(len(model_names)))
        axes[1,1].set_xticklabels(model_names, rotation=45)
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_final_models(self):
        """
        Finalna evaluacija najboljih modela na test skupu
        """
        print("\n" + "="*50)
        print("🎯 FINALNA EVALUACIJA NA TEST SKUPU")
        print("="*50)
        
        final_results = {}
        
        for name, result in self.results.items():
            print(f"\n📊 Evaluacija {name}...")
            
            # Odabir boljeg modela (original vs PCA)
            if result['cv_r2_original'] > result['cv_r2_pca']:
                best_model = result['model_original']
                X_test_use = self.X_test_scaled
                space_type = "Original"
                cv_r2 = result['cv_r2_original']
            else:
                best_model = result['model_pca']
                X_test_use = self.X_test_pca
                space_type = "PCA"
                cv_r2 = result['cv_r2_pca']
            
            # Predikcija na test skupu
            y_pred = best_model.predict(X_test_use)
            
            # Računanje mera uspešnosti
            mae = mean_absolute_error(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)
            
            # Procenat predikcija u ±10% od tačne vrednosti
            percentage_error = np.abs((y_pred - self.y_test) / self.y_test) * 100
            within_10_percent = np.sum(percentage_error <= 10) / len(percentage_error) * 100
            within_20_percent = np.sum(percentage_error <= 20) / len(percentage_error) * 100
            
            final_results[name] = {
                'model': best_model,
                'space_type': space_type,
                'cv_r2': cv_r2,
                'test_mae': mae,
                'test_mse': mse,
                'test_rmse': rmse,
                'test_r2': r2,
                'within_10_percent': within_10_percent,
                'within_20_percent': within_20_percent,
                'y_pred': y_pred
            }
            
            print(f"  🏆 Najbolji prostor: {space_type}")
            print(f"  📈 Test MAE: ${mae:,.0f}")
            print(f"  📈 Test MSE: {mse:,.0f}")
            print(f"  📈 Test RMSE: ${rmse:,.0f}")
            print(f"  📊 Test R²: {r2:.3f}")
            print(f"  🎯 Predikcije u ±10%: {within_10_percent:.1f}%")
            print(f"  🎯 Predikcije u ±20%: {within_20_percent:.1f}%")
        
        # Tabela rezultata
        self.display_results_table(final_results)
        
        # Vizuelizacija rezultata
        self.plot_final_results(final_results)
        
        # Analiza grešaka za najbolji model
        best_model_name = max(final_results.keys(), key=lambda x: final_results[x]['test_r2'])
        self.analyze_predictions(final_results[best_model_name], best_model_name)
        
        return final_results
    
    def display_results_table(self, final_results):
        """
        Prikazuje tabelu sa svim rezultatima
        """
        print(f"\n{'='*80}")
        print("📋 TABELA SVIH REZULTATA")
        print(f"{'='*80}")
        
        # Header
        print(f"{'Model':<15} {'Prostor':<8} {'CV R²':<8} {'Test R²':<8} {'RMSE':<12} {'MAE':<12} {'±10%':<6} {'±20%':<6}")
        print(f"{'-'*80}")
        
        # Sortiranje po test R²
        sorted_results = sorted(final_results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
        
        for name, result in sorted_results:
            print(f"{name:<15} {result['space_type']:<8} {result['cv_r2']:<8.3f} "
                  f"{result['test_r2']:<8.3f} ${result['test_rmse']:<11,.0f} "
                  f"${result['test_mae']:<11,.0f} {result['within_10_percent']:<5.1f}% "
                  f"{result['within_20_percent']:<5.1f}%")
        
        print(f"{'-'*80}")
        
        # Najbolji model
        best_model = sorted_results[0]
        print(f"🏆 NAJBOLJI MODEL: {best_model[0]} ({best_model[1]['space_type']} prostor)")
        print(f"   Test R²: {best_model[1]['test_r2']:.3f}")
        print(f"   RMSE: ${best_model[1]['test_rmse']:,.0f}")
    
    def plot_final_results(self, final_results):
        """
        Vizuelizuje finalne rezultate
        """
        model_names = list(final_results.keys())
        test_r2 = [final_results[name]['test_r2'] for name in model_names]
        test_rmse = [final_results[name]['test_rmse'] for name in model_names]
        within_10 = [final_results[name]['within_10_percent'] for name in model_names]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # R² score
        colors = ['gold' if r2 == max(test_r2) else 'skyblue' for r2 in test_r2]
        bars1 = axes[0,0].bar(model_names, test_r2, color=colors, alpha=0.8)
        axes[0,0].set_ylabel('Test R² Score')
        axes[0,0].set_title('Test R² Score po modelima')
        axes[0,0].set_ylim(0, 1)
        axes[0,0].grid(axis='y', alpha=0.3)
        
        # Dodavanje vrednosti na barove
        for bar, r2 in zip(bars1, test_r2):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # RMSE
        colors = ['lightcoral' if rmse == min(test_rmse) else 'lightblue' for rmse in test_rmse]
        bars2 = axes[0,1].bar(model_names, test_rmse, color=colors, alpha=0.8)
        axes[0,1].set_ylabel('Test RMSE ($)')
        axes[0,1].set_title('Test RMSE po modelima')
        axes[0,1].grid(axis='y', alpha=0.3)
        
        # Dodavanje vrednosti na barove
        for bar, rmse in zip(bars2, test_rmse):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(test_rmse)*0.01, 
                          f'${rmse:,.0f}', ha='center', va='bottom', fontweight='bold', rotation=90)
        
        # Procenat tačnih predikcija
        bars3 = axes[1,0].bar(model_names, within_10, color='lightgreen', alpha=0.8)
        axes[1,0].set_ylabel('Procenat (%)')
        axes[1,0].set_title('Predikcije u ±10% od tačne vrednosti')
        axes[1,0].set_ylim(0, 100)
        axes[1,0].grid(axis='y', alpha=0.3)
        
        for bar, pct in zip(bars3, within_10):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                          f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Poređenje CV vs Test R²
        cv_r2 = [final_results[name]['cv_r2'] for name in model_names]
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[1,1].bar(x - width/2, cv_r2, width, label='CV R²', alpha=0.8, color='orange')
        axes[1,1].bar(x + width/2, test_r2, width, label='Test R²', alpha=0.8, color='purple')
        axes[1,1].set_xlabel('Modeli')
        axes[1,1].set_ylabel('R² Score')
        axes[1,1].set_title('CV vs Test R² poređenje')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(model_names, rotation=45)
        axes[1,1].legend()
        axes[1,1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_predictions(self, best_result, model_name):
        """
        Detaljana analiza predikcija najboljeg modela
        """
        print(f"\n{'='*60}")
        print(f"🔍 ANALIZA PREDIKCIJA - {model_name}")
        print(f"{'='*60}")
        
        y_true = self.y_test
        y_pred = best_result['y_pred']
        
        # Residuali
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Actual vs Predicted
        axes[0,0].scatter(y_true, y_pred, alpha=0.6, s=20)
        min_price = min(y_true.min(), y_pred.min())
        max_price = max(y_true.max(), y_pred.max())
        axes[0,0].plot([min_price, max_price], [min_price, max_price], 'r--', lw=2)
        axes[0,0].set_xlabel('Stvarna cena ($)')
        axes[0,0].set_ylabel('Predviđena cena ($)')
        axes[0,0].set_title('Stvarna vs Predviđena cena')
        axes[0,0].grid(True, alpha=0.3)
        
        # R² na grafiku
        r2_text = f'R² = {best_result["test_r2"]:.3f}'
        axes[0,0].text(0.05, 0.95, r2_text, transform=axes[0,0].transAxes, 
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                      fontsize=12, fontweight='bold')
        
        # 2. Residuali vs Predicted
        axes[0,1].scatter(y_pred, residuals, alpha=0.6, s=20)
        axes[0,1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0,1].set_xlabel('Predviđena cena ($)')
        axes[0,1].set_ylabel('Residuali ($)')
        axes[0,1].set_title('Residuali vs Predviđena cena')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Distribucija residuala
        axes[0,2].hist(residuals, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0,2].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[0,2].set_xlabel('Residuali ($)')
        axes[0,2].set_ylabel('Frekvencija')
        axes[0,2].set_title('Distribucija residuala')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Q-Q plot residuala
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1,0])
        axes[1,0].set_title('Q-Q plot residuala')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Procenat greške po cenovnim opsezima
        price_ranges = [(0, 300000), (300000, 500000), (500000, 800000), (800000, 1500000), (1500000, float('inf'))]
        range_labels = ['<$300k', '$300k-500k', '$500k-800k', '$800k-1.5M', '>$1.5M']
        range_accuracies = []
        
        for low, high in price_ranges:
            mask = (y_true >= low) & (y_true < high)
            if mask.sum() > 0:
                range_errors = np.abs((y_pred[mask] - y_true[mask]) / y_true[mask]) * 100
                within_10 = (range_errors <= 10).mean() * 100
                range_accuracies.append(within_10)
            else:
                range_accuracies.append(0)
        
        bars = axes[1,1].bar(range_labels, range_accuracies, color='lightgreen', alpha=0.8)
        axes[1,1].set_ylabel('Predikcije u ±10% (%)')
        axes[1,1].set_title('Tačnost po cenovnim opsezima')
        axes[1,1].set_xticklabels(range_labels, rotation=45)
        axes[1,1].grid(axis='y', alpha=0.3)
        
        # Dodavanje vrednosti na barove
        for bar, acc in zip(bars, range_accuracies):
            if acc > 0:
                axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                              f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 6. Distribucija apsolutne greške u procentima
        abs_pct_error = np.abs((y_pred - y_true) / y_true) * 100
        axes[1,2].hist(abs_pct_error, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1,2].axvline(x=10, color='r', linestyle='--', lw=2, label='10%')
        axes[1,2].axvline(x=20, color='g', linestyle='--', lw=2, label='20%')
        axes[1,2].set_xlabel('Apsolutna greška (%)')
        axes[1,2].set_ylabel('Frekvencija')
        axes[1,2].set_title('Distribucija apsolutne greške')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Statistike grešaka
        print(f"\n📊 STATISTIKE GREŠAKA:")
        print(f"Srednja apsolutna greška: ${np.mean(np.abs(residuals)):,.0f}")
        print(f"Medijana apsolutne greške: ${np.median(np.abs(residuals)):,.0f}")
        print(f"Standardna devijacija residuala: ${np.std(residuals):,.0f}")
        print(f"Maksimalna pozitivna greška: ${residuals.max():,.0f}")
        print(f"Maksimalna negativna greška: ${residuals.min():,.0f}")
        
        print(f"\n🎯 PROCENAT TAČNIH PREDIKCIJA:")
        for threshold in [5, 10, 15, 20, 25]:
            within_threshold = (abs_pct_error <= threshold).mean() * 100
            print(f"U ±{threshold}%: {within_threshold:.1f}%")
        
        # Identifikacija najgorih predikcija
        worst_predictions_idx = np.argsort(abs_pct_error)[-5:]
        print(f"\n❌ 5 NAJGORIH PREDIKCIJA:")
        print(f"{'Stvarna':<12} {'Predviđena':<12} {'Greška':<10} {'Greška %':<10}")
        print("-" * 50)
        for idx in worst_predictions_idx[::-1]:
            actual = y_true.iloc[idx]
            predicted = y_pred[idx]
            error = predicted - actual
            error_pct = abs_pct_error[idx]
            print(f"${actual:<11,.0f} ${predicted:<11,.0f} ${error:<9,.0f} {error_pct:<9.1f}%")
    
    def run_complete_analysis(self):
        """
        Pokreće kompletnu analizu
        """
        print("🚀 POKRETANJE KOMPLETNE ANALIZE PREDVIĐANJA CENA NEKRETNINA")
        print("=" * 80)
        
        # 1. Eksplorativna analiza
        self.explore_data()
        
        # 2. Priprema podataka
        self.preprocess_data()
        
        # 3. PCA
        self.apply_pca()
        
        # 4. Treniranje modela
        self.train_models()
        
        # 5. Finalna evaluacija
        final_results = self.evaluate_final_models()
        
        print(f"\n{'='*80}")
        print("✅ ANALIZA ZAVRŠENA!")
        print(f"{'='*80}")
        
        # Najbolji model
        best_model_name = max(final_results.keys(), key=lambda x: final_results[x]['test_r2'])
        best_r2 = final_results[best_model_name]['test_r2']
        best_rmse = final_results[best_model_name]['test_rmse']
        
        print(f"🏆 NAJBOLJI MODEL: {best_model_name}")
        print(f"📊 Test R²: {best_r2:.3f}")
        print(f"📈 Test RMSE: ${best_rmse:,.0f}")
        print(f"🎯 Prostor: {final_results[best_model_name]['space_type']}")
        
        return final_results


# UPUTSTVO ZA POKRETANJE ANALIZE
def main():
    """
    Glavna funkcija za pokretanje analize
    """
    print("🏠 PROJEKAT: PREDVIĐANJE CENA NEKRETNINA")
    print("=" * 50)
    
    # OVDE UNESITE PUTANJU DO VAŠEG CSV FAJLA
    csv_path = "dataset/kc_house_data.csv"  # Zamenite sa putanjom do vašeg fajla
    
    try:
        # Kreiranje instance klase
        analyzer = HousePricePrediction(csv_path)
        
        # Pokretanje kompletne analize
        results = analyzer.run_complete_analysis()
        
        print("\n🎉 Analiza je uspešno završena!")
        print("📋 Rezultati su sačuvani u 'results' varijabli.")
        
        return analyzer, results
        
    except Exception as e:
        print(f"❌ Greška: {e}")
        print("\n💡 SAVETI:")
        print("1. Proverite da li je putanja do CSV fajla tačna")
        print("2. Proverite da li CSV fajl sadrži sve potrebne kolone")
        print("3. Proverite da li su instalirane sve potrebne biblioteke")
        
        return None, None


# POKRETANJE ANALIZE
if __name__ == "__main__":
    # Pokretanje glavne funkcije
    analyzer, results = main()
    
    # Dodatne analize (opciono)
    if analyzer is not None:
        print("\n" + "="*50)
        print("💡 DODATNE MOGUĆNOSTI:")
        print("="*50)
        print("analyzer.explore_data()           # Ponovni prikaz EDA")
        print("analyzer.results                  # CV rezultati")
        print("results                          # Finalni rezultati")
        print("analyzer.data.head()             # Pregled podataka")
        print("analyzer.data.describe()         # Statistike podataka")
# %%

# %%
