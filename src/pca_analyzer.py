# ===================================================================
# MODUL ZA PCA ANALIZU
# ===================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def apply_pca(X_train_scaled, X_test_scaled, X_train, n_components=0.95):
    """
    Primenjuje PCA za smanjenje dimenzionalnosti
    """
    print("\n" + "="*50)
    print("🔍 PRIMENA PCA")
    print("="*50)
    
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"📊 Originalan broj feature-a: {X_train_scaled.shape[1]}")
    print(f"🎯 Broj komponenti nakon PCA: {X_train_pca.shape[1]}")
    print(f"📈 Objašnjena varijansa: {pca.explained_variance_ratio_.sum():.3f} ({pca.explained_variance_ratio_.sum()*100:.1f}%)")
    
    # Vizuelizacija objašnjene varijanse
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_, alpha=0.7, color='skyblue')
    plt.xlabel('Komponenta')
    plt.ylabel('Objašnjena varijansa')
    plt.title('Objašnjena varijansa po komponenti')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 2, 2)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
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
    feature_names = X_train.columns
    print(f"\n🏆 NAJVAŽNIJI FEATURE-I ZA PRVE 3 KOMPONENTE:")
    
    for i in range(min(3, pca.n_components_)):
        component = pca.components_[i]
        top_features_idx = np.argsort(np.abs(component))[-5:][::-1]
        
        print(f"\nKomponenta {i+1}:")
        for j, idx in enumerate(top_features_idx):
            print(f"  {j+1}. {feature_names[idx]}: {component[idx]:.3f}")
    
    return X_train_pca, X_test_pca, pca
