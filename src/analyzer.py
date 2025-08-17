import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def analyze_predictions(best_result, model_name, y_test):
    """
    Detaljana analiza predikcija najboljeg modela
    """
    # Koristi striktno poziciono indeksiranje preko NumPy da bi se izbegli KeyError-i
    y_true = y_test  # zadržavamo Series za kompatibilnost gde je zgodno
    y_true_np = np.asarray(y_test)
    y_pred = np.asarray(best_result['y_pred'])
    
    # Residuali
    residuals = y_true_np - y_pred
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Actual vs Predicted
    axes[0,0].scatter(y_true_np, y_pred, alpha=0.6, s=20)
    min_price = min(y_true_np.min(), y_pred.min())
    max_price = max(y_true_np.max(), y_pred.max())
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
    stats.probplot(residuals, dist="norm", plot=axes[1,0])
    axes[1,0].set_title('Q-Q plot residuala')
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Procenat greške po cenovnim opsezima
    price_ranges = [(0, 300000), (300000, 500000), (500000, 800000), (800000, 1500000), (1500000, float('inf'))]
    range_labels = ['<$300k', '$300k-500k', '$500k-800k', '$800k-1.5M', '>$1.5M']
    range_accuracies = []
    
    for low, high in price_ranges:
        mask = (y_true_np >= low) & (y_true_np < high)
        if mask.sum() > 0:
            range_errors = np.abs((y_pred[mask] - y_true_np[mask]) / y_true_np[mask]) * 100
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
    abs_pct_error = np.abs((y_pred - y_true_np) / y_true_np) * 100
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
    print(f"\nSTATISTIKE GREŠAKA:")
    print(f"Srednja apsolutna greška: ${np.mean(np.abs(residuals)):,.0f}")
    print(f"Medijana apsolutne greške: ${np.median(np.abs(residuals)):,.0f}")
    print(f"Standardna devijacija residuala: ${np.std(residuals):,.0f}")
    print(f"Maksimalna pozitivna greška: ${residuals.max():,.0f}")
    print(f"Maksimalna negativna greška: ${residuals.min():,.0f}")
    
    print(f"\nPROCENAT TAČNIH PREDIKCIJA:")
    for threshold in [5, 10, 15, 20, 25]:
        within_threshold = (abs_pct_error <= threshold).mean() * 100
        print(f"U ±{threshold}%: {within_threshold:.1f}%")
    
    # Identifikacija najgorih predikcija
    worst_predictions_idx = np.argsort(abs_pct_error)[-5:]
    print(f"\n5 NAJGORIH PREDIKCIJA:")
    print(f"{'Stvarna':<12} {'Predviđena':<12} {'Greška':<10} {'Greška %':<10}")
    print("-" * 50)
    for idx in worst_predictions_idx[::-1]:
        actual = y_true_np[idx]
        predicted = y_pred[idx]
        error = predicted - actual
        error_pct = abs_pct_error[idx]
        print(f"${actual:<11,.0f} ${predicted:<11,.0f} ${error:<9,.0f} {error_pct:<9.1f}%")
