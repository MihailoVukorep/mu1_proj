import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV

def train_models(X_train_scaled, X_train_pca, y_train):
    """
    Trenira različite modele sa cross-validation i hyperparameter tuning
    """
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
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTreniranje {name}...")
        
        # Grid Search sa Cross Validation za originalne podatke
        if param_grids[name]:
            print("  🔍 Pretraga hiperparametara (originalni prostor)...")
            grid_search_orig = GridSearchCV(
                model, param_grids[name], 
                cv=5, scoring='neg_mean_squared_error', 
                n_jobs=-1, verbose=0
            )
            grid_search_orig.fit(X_train_scaled, y_train)
            best_model_orig = grid_search_orig.best_estimator_
            best_params_orig = grid_search_orig.best_params_
            
            print("  🔍 Pretraga hiperparametara (PCA prostor)...")
            grid_search_pca = GridSearchCV(
                model, param_grids[name], 
                cv=5, scoring='neg_mean_squared_error', 
                n_jobs=-1, verbose=0
            )
            grid_search_pca.fit(X_train_pca, y_train)
            best_model_pca = grid_search_pca.best_estimator_
            best_params_pca = grid_search_pca.best_params_
            
        else:
            # Za Linear Regression nema hiperparametara
            best_model_orig = model
            best_model_orig.fit(X_train_scaled, y_train)
            best_params_orig = {}
            
            best_model_pca = LinearRegression()
            best_model_pca.fit(X_train_pca, y_train)
            best_params_pca = {}
        
        # Cross-validation rezultati
        print("Cross-validation evaluacija...")
        cv_scores_orig = cross_val_score(
            best_model_orig, X_train_scaled, y_train, 
            cv=5, scoring='neg_mean_squared_error'
        )
        cv_scores_pca = cross_val_score(
            best_model_pca, X_train_pca, y_train, 
            cv=5, scoring='neg_mean_squared_error'
        )
        
        # R2 score cross-validation
        cv_r2_orig = cross_val_score(
            best_model_orig, X_train_scaled, y_train, 
            cv=5, scoring='r2'
        )
        cv_r2_pca = cross_val_score(
            best_model_pca, X_train_pca, y_train, 
            cv=5, scoring='r2'
        )
        
        results[name] = {
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
        
        print(f"  {name} - Završeno!")
        print(f"     CV MSE (Original): {results[name]['cv_mse_original']:,.0f} ± {results[name]['cv_mse_std_original']:,.0f}")
        print(f"     CV MSE (PCA): {results[name]['cv_mse_pca']:,.0f} ± {results[name]['cv_mse_std_pca']:,.0f}")
        print(f"     CV R² (Original): {results[name]['cv_r2_original']:.3f} ± {results[name]['cv_r2_std_original']:.3f}")
        print(f"     CV R² (PCA): {results[name]['cv_r2_pca']:.3f} ± {results[name]['cv_r2_std_pca']:.3f}")
    
    return results

def plot_cv_results(results):
    """
    Vizuelizuje cross-validation rezultate
    """
    model_names = list(results.keys())
    cv_mse_orig = [results[name]['cv_mse_original'] for name in model_names]
    cv_mse_pca = [results[name]['cv_mse_pca'] for name in model_names]
    cv_r2_orig = [results[name]['cv_r2_original'] for name in model_names]
    cv_r2_pca = [results[name]['cv_r2_pca'] for name in model_names]
    
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
    mse_orig_std = [results[name]['cv_mse_std_original'] for name in model_names]
    mse_pca_std = [results[name]['cv_mse_std_pca'] for name in model_names]
    
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
    r2_orig_std = [results[name]['cv_r2_std_original'] for name in model_names]
    r2_pca_std = [results[name]['cv_r2_std_pca'] for name in model_names]
    
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
