import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_final_models(results, X_test_scaled, X_test_pca, y_test):
    """
    Finalna evaluacija najboljih modela na test skupu
    """
    print("\n" + "="*50)
    print("🎯 FINALNA EVALUACIJA NA TEST SKUPU")
    print("="*50)
    
    final_results = {}
    
    for name, result in results.items():
        print(f"\n📊 Evaluacija {name}...")
        
        # Odabir boljeg modela (original vs PCA)
        if result['cv_r2_original'] > result['cv_r2_pca']:
            best_model = result['model_original']
            X_test_use = X_test_scaled
            space_type = "Original"
            cv_r2 = result['cv_r2_original']
        else:
            best_model = result['model_pca']
            X_test_use = X_test_pca
            space_type = "PCA"
            cv_r2 = result['cv_r2_pca']
        
        # Predikcija na test skupu
        y_pred = best_model.predict(X_test_use)
        
        # Računanje mera uspešnosti
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Procenat predikcija u ±10% od tačne vrednosti
        percentage_error = np.abs((y_pred - y_test) / y_test) * 100
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
    
    return final_results

def display_results_table(final_results):
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

def plot_final_results(final_results):
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
