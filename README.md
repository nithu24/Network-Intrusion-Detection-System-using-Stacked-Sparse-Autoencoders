# 🔐 Network Intrusion Detection System using Stacked Sparse Autoencoders

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![LightGBM](https://img.shields.io/badge/LightGBM-Classifier-green)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-blue)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-purple)
![Dataset](https://img.shields.io/badge/Dataset-NSL--KDD-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> **A deep learning-powered Network Intrusion Detection System (IDS) using Stacked Sparse Autoencoders for feature extraction, combined with LightGBM and XGBoost classifiers — evaluated across Binary, 5-class, and 13-class attack classification tasks.**

---

## 🎯 Problem Statement

Network intrusion detection is a critical cybersecurity challenge. Traditional ML approaches struggle with high-dimensional network traffic data and subtle attack patterns. This project investigates whether **deep unsupervised feature extraction via Stacked Sparse Autoencoders** can improve classification performance over raw features, across three levels of attack granularity.

**Research Questions:**
- Does stacked autoencoder feature extraction improve intrusion detection accuracy?
- How does performance compare across Binary, 5-class, and 13-class classification tasks?
- Which classifier (LightGBM vs XGBoost) benefits more from autoencoder-extracted features?
- What is the optimal sparsity target for the KL-divergence regularization?

---

## 📁 Project Structure

```
network-intrusion-detection/
│
├── 📂 Binary Classification (Normal vs Attack)
│   ├── Binary_XGBoost_Gridsearchcv_Model.ipynb
│   ├── Alone_Binary_LGBM_Model.ipynb
│   ├── Binary_Stacked_Autoencoders_LGBM_Optuna_Sparse_target_0_01.ipynb
│   ├── Binary_Stacked_Autoencoders_LGBM_Optuna_Sparse_target_0_04.ipynb
│   ├── Binary_Stacked_Autoencoders_LGBM_Optuna_Sparse_target_0_05.ipynb
│   └── Binary_Stacked_Autoencoders_LGBM_Optuna_Sparse_target_0_06.ipynb
│
├── 📂 Five-Class Classification (Normal + 4 Attack Categories)
│   ├── Five_SA_XGBoost_Gridsearchcv_Model.ipynb
│   ├── Alone__Five_-_LGBM__Model.ipynb
│   ├── Five_Stacked_Autoencoders_LGBM_Gridsearchcv_Model_Sparse_target_0_01.ipynb
│   ├── Five_Stacked_Autoencoders_LGBM_Gridsearchcv_Model_Sparse_target_0_04.ipynb
│   ├── Five_Stacked_Autoencoders__LGBM_Gridsearchcv_Model_Sparse_target_0_06.ipynb
│   └── Five_Stacked_Autoencoders_LGBM_Best_Parameter_Model_Sparse_target_0_05.ipynb
│
├── 📂 Thirteen-Class Classification (Fine-grained Attack Types)
│   ├── Thirteen_SA_XGBoost_Gridsearchcv_Model.ipynb
│   └── Thirteen_Stacked_Autoencoders_LGBM_Optuna_Model.ipynb
│
├── KDDTrain+.txt                           # NSL-KDD Training Dataset
└── README.md
```

---

## 📊 Dataset — NSL-KDD

| Property | Details |
|----------|---------|
| **Dataset** | NSL-KDD (improved version of KDD Cup 1999) |
| **File** | `KDDTrain+.txt` |
| **Features** | 41 network traffic features (41 numerical + categorical) |
| **Target** | Attack type label (41 unique attack names + normal) |
| **Task Variants** | Binary / 5-class / 13-class classification |

### Attack Categories

**Binary:** Normal vs Attack (all attacks grouped)

**5-Class:** Normal, DoS, Probe, R2L, U2R

**13-Class:** Normal + 12 specific attack types (after removing rare classes with <10 samples: ftp_write, multihop, rootkit, imap, warezmaster, phf, land, loadmodule, spy, perl)

---

## 🏗️ Architecture — Stacked Sparse Autoencoders

### What Are Stacked Sparse Autoencoders?

A **Stacked Autoencoder** chains multiple autoencoders sequentially — the compressed output (bottleneck) of one autoencoder becomes the input to the next. This progressively extracts increasingly abstract feature representations.

**Sparsity** is enforced using **KL-Divergence regularization**, encouraging most neurons to remain inactive — learning a compact, meaningful representation of network traffic patterns.

```
Raw Features (122 dims)
        ↓
Autoencoder 1: 122 → 64 (Bottleneck)
        ↓
Autoencoder 2: 64 → 32 (Bottleneck)
        ↓
Autoencoder 3: 32 → 16 (Bottleneck)
        ↓
Autoencoder 4: 16 → 8  (Bottleneck)
        ↓
Compressed Features (8 dims)
        ↓
LightGBM / XGBoost Classifier
```

### Autoencoder Configuration

```python
bottleneck_dims    = [64, 32, 16, 8]
learning_rate      = 0.005
weight_decay       = 0.001
sparsity_weight    = 0.1
epochs             = 10
batch_size         = 256
activation         = 'relu' (encoder) / 'sigmoid' (decoder)
optimizer          = Adam
```

### Custom Sparse Loss Function

```python
def custom_loss(y_true, y_pred):
    mse = MeanSquaredError()(y_true, y_pred)
    avg_activation = K.mean(y_pred, axis=0)
    kl = K.sum(
        sparsity_target * K.log(sparsity_target / avg_activation) +
        (1 - sparsity_target) * K.log((1 - sparsity_target) / (1 - avg_activation))
    )
    return mse + sparsity_weight * kl
```

> **MSE** ensures accurate reconstruction. **KL Divergence** enforces sparsity by penalizing deviations from the target activation rate.

---

## ⚙️ Preprocessing Pipeline

```python
# MinMaxScaler for numerical features
# OneHotEncoder for categorical features (protocol_type, service, flag)
# ColumnTransformer combining both

preprocessor = ColumnTransformer(transformers=[
    ('num', MinMaxScaler(), numerical_cols),
    ('cat', OneHotEncoder(), categorical_cols)
])
```

- Removed `customerID`-equivalent column (column 42)
- Label encoded target variable for multi-class tasks
- For 13-class: removed 10 rare attack classes with insufficient samples

---

## 🧪 Experiments — Sparsity Target Ablation Study

Multiple sparsity targets tested to find optimal KL-divergence constraint:

| Sparsity Target | Notebooks |
|----------------|-----------|
| 0.01 | Binary, Five-class |
| 0.04 | Binary, Five-class |
| **0.05** | Binary, Five-class (**selected as best**) |
| 0.06 | Binary, Five-class |

---

## 🤖 Classifiers & Hyperparameter Tuning

### LightGBM + Optuna (Binary & 13-class)
```python
# Optuna searches over:
# boosting_type: gbdt / dart / rf
# learning_rate: 0.001 – 0.5 (log scale)
# num_leaves: 2 – 1000
# max_depth: 3 – 20
# subsample, bagging_freq, min_data_in_leaf, colsample_bytree

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
```

### LightGBM + GridSearchCV (5-class)

### XGBoost + GridSearchCV (Binary, 5-class, 13-class)
```python
param_grid = {
    'max_depth': [6, 8, 10],
    'learning_rate': [0.1, 0.3],
    'n_estimators': [100, 200, 300],
    'subsample': [0.5, 0.8, 1.0]
}
```

---

## 📈 Evaluation Metrics

Each experiment evaluated on:
- **Accuracy**
- **Precision, Recall, F1-Score** (per class + weighted)
- **ROC-AUC Score**
- **Confusion Matrix** (heatmap)
- **ROC Curve** (with AUC)
- **Train vs Test metrics** (overfitting check)

---

## 🔬 Experimental Design Summary

| Experiment | Feature Source | Classifier | Tuning |
|-----------|---------------|-----------|--------|
| Baseline Binary | Raw features | LGBM | Optuna |
| SA Binary | Autoencoder (sparse=0.01–0.06) | LGBM | Optuna |
| SA Binary | Autoencoder | XGBoost | GridSearchCV |
| Baseline 5-class | Raw features | LGBM | Optuna |
| SA 5-class | Autoencoder (sparse=0.01–0.06) | LGBM | GridSearchCV |
| SA 5-class | Autoencoder | XGBoost | GridSearchCV |
| SA 13-class | Autoencoder | LGBM | Optuna |
| SA 13-class | Autoencoder | XGBoost | GridSearchCV |

---

## 🛠️ Tools & Technologies

| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| Deep Learning | TensorFlow 2.x, Keras |
| ML Classifiers | LightGBM, XGBoost, Scikit-learn |
| Hyperparameter Tuning | Optuna (Bayesian), GridSearchCV |
| Preprocessing | Scikit-learn (MinMaxScaler, OneHotEncoder, LabelEncoder, ColumnTransformer) |
| Visualization | Matplotlib, Seaborn |
| Model Saving | Joblib |
| Dataset | NSL-KDD (KDDTrain+.txt) |

---


## 📦 Requirements

```
tensorflow>=2.x
lightgbm
xgboost
optuna
scikit-learn
pandas
numpy
matplotlib
seaborn
joblib
```

---

## 👩‍💻 Author

**Nithu Anna Ninan**
- 🔗 LinkedIn: [linkedin.com/in/nithu-ninan](https://linkedin.com/in/nithu-ninan)
- 📧 Email: nithuanna24@gmail.com

---

## ⭐ If you found this project useful, please give it a star!
