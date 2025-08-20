# 🏠 House Price Prediction ML Project
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

## 🎯 Project Overview
This project develops and compares multiple machine learning models to predict house prices in King County, Washington. The analysis includes comprehensive exploratory data analysis (EDA), feature engineering, dimensionality reduction using Principal Component Analysis (PCA), and model evaluation with cross-validation.

### Key Objectives:
- 🔍 Perform thorough exploratory data analysis
- 🛠️ Engineer meaningful features from raw data
- 📉 Apply PCA for dimensionality reduction
- 🤖 Train and compare multiple ML algorithms
- 📊 Evaluate model performance with comprehensive metrics
- 🎨 Create insightful visualizations

## 📊 Dataset
- **Source**: [King County House Sales Dataset](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
- **Location**: `dataset/kc_house_data.csv`
- **Size**: 21,613 records with 21 features

### Key Features:
- `price`: Target variable (house price in USD)
- `sqft_living`: Living space square footage
- `bedrooms`: Number of bedrooms
- `bathrooms`: Number of bathrooms
- `grade`: Overall grade of the house (1-13)
- `waterfront`: Waterfront property (0/1)
- `view`: View rating (0-4)
- `condition`: Condition rating (1-5)
- `sqft_lot`: Lot size square footage
- `floors`: Number of floors
- `yr_built`: Year built
- `yr_renovated`: Year renovated
- `lat`, `long`: Geographic coordinates

## 🔧 Features

### 🔬 Advanced Analytics
- **Comprehensive EDA**: 9-panel visualization dashboard
- **Feature Engineering**: Creates new meaningful features
- **Correlation Analysis**: Feature selection based on correlation thresholds
- **PCA Implementation**: Dimensionality reduction with variance explanation
- **Cross-Validation**: 5-fold CV for robust model evaluation

### 🤖 Machine Learning Models
- **Linear Regression**: Baseline linear model
- **Random Forest**: Ensemble tree-based method
- **Support Vector Regression (SVR)**: Non-linear regression
- **K-Nearest Neighbors (KNN)**: Instance-based learning

### 📊 Evaluation Metrics
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **Prediction Accuracy**: Percentage within ±10% and ±20% of actual values

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/MihailoVukorep/mu1_proj
cd mu1_proj

# Set up virtual environment (Linux/Mac)
chmod +x setup-env.sh
./setup-env.sh

# Or install dependencies manually
pip install -r requirements.txt

# Use Jupyter notebook for interactive analysis
main.ipynb
```

## 📁 Project Structure

```
mu1_proj/
├── 📓 main.ipynb              # Interactive Jupyter notebook
├── 📊 stats.py                # Statistical analysis utilities
├── 📋 requirements.txt        # Python dependencies
├── 🔧 setup-env.sh           # Environment setup script
├── 📁 dataset/
│   ├── kc_house_data.csv     # Main dataset
│   └── kc_house_data.csv.zip # Compressed dataset
├── 📁 src/                   # Source code modules
│   ├── __init__.py
│   ├── 📥 data_loader.py     # Data loading and EDA
│   ├── 🔧 preprocessor.py    # Data preprocessing
│   ├── 📉 pca_analyzer.py    # PCA implementation
│   ├── 🤖 model_trainer.py   # Model training
│   ├── 📊 evaluator.py       # Model evaluation
│   └── 🔍 analyzer.py        # Prediction analysis
├── 📁 docs/
│   └── deo_1.txt            # Documentation
```

## 🔬 Methodology

### 1. Data Preprocessing
- **Data Cleaning**: Handle missing values and outliers
- **Feature Engineering**: Create new meaningful features:
  - `house_age`: Current year - year built
  - `is_renovated`: Binary renovation indicator
  - `basement_ratio`: Basement area ratio
  - `is_large_house`: Above-average size indicator
  - `luxury_score`: Combined luxury features score

### 2. Feature Selection
- Correlation-based feature selection (threshold: 0.1)
- Remove highly correlated and low-impact features

### 3. Dimensionality Reduction
- PCA with 95% variance retention
- Compare original vs. PCA-transformed features

### 4. Model Training & Evaluation
- **Hyperparameter Tuning**: GridSearchCV for optimal parameters
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Model Comparison**: Evaluate on both original and PCA features

## 👥 Authors

- **Mihailo Vukorep** - *IN 40/2021* - [GitHub](https://github.com/MihailoVukorep)
- **Marko Kolarski** - *IN 60/2021* - [GitHub](https://github.com/MarkoKolarski)
