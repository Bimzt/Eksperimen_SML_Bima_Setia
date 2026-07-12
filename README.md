# Eksperimen_SML_Bima_Setia

# Heart Disease Prediction ML System Experiment

> End-to-end Machine Learning pipeline for heart disease classification, built with automated retraining via MLflow Projects + GitHub Actions.

---

## Overview

Proyek ini membangun sistem ML yang dapat memprediksi risiko penyakit jantung berdasarkan data klinis pasien. Lebih dari sekadar notebook eksplorasi, pipeline ini dirancang untuk dapat dijalankan ulang secara otomatis — mencakup preprocessing terstruktur, eksperimen tracking dengan MLflow, hingga retraining terjadwal via GitHub Actions.

**Dataset:** [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

---

## Results

| Metric | Value |
|--------|-------|
| Model | Random Forest Classifier |
| Accuracy | **80%** |
| Train/Test Split | 80:20 |

---

## Pipeline Architecture

```
heart_raw/
    └── raw CSV dataset (Kaggle)
            │
            ▼
    preprocessing/
            │
            ├── Handling missing values
            ├── Categorical encoding
            ├── Feature scaling (StandardScaler)
            └── Train-test split (80:20)
                        │
                        ▼
            Model Training
                        │
                        ├── Random Forest Classifier
                        └── MLflow experiment tracking
                                    │
                                    ▼
                    .github/workflows/
                            │
                            └── GitHub Actions — automated retraining
```

---

## Repository Structure

```
├── heart_raw/                        # Raw dataset from Kaggle
├── preprocessing/                    # Preprocessing notebooks & scripts
│   ├── preprocessing.ipynb           # Full preprocessing pipeline
│   └── ...
├── .github/
│   └── workflows/
│       └── retrain.yml               # Automated retraining workflow
└── README.md
```

---

## Preprocessing Steps

1. **Handling Missing Values** — identifikasi dan imputasi nilai kosong pada fitur klinis
2. **Categorical Encoding** — konversi fitur kategorikal ke representasi numerik
3. **Feature Scaling** — normalisasi fitur numerik dengan StandardScaler
4. **Train-Test Split** — pembagian dataset 80% training / 20% testing dengan stratified sampling

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python |
| ML Framework | Scikit-learn |
| Experiment Tracking | MLflow |
| Automation | GitHub Actions |
| Notebook | Jupyter Notebook |
| Dataset Source | Kaggle |

---

## Automated Retraining

Pipeline ini dilengkapi GitHub Actions workflow yang memungkinkan retraining model secara otomatis. MLflow digunakan untuk mencatat setiap eksperimen — parameter, metrik, dan artefak model — sehingga setiap run dapat dibandingkan dan direproduksi.

---

## Getting Started

### 1. Clone repository

```bash
git clone https://github.com/Bimzt/Eksperimen_SML_Bima_Setia.git
cd Eksperimen_SML_Bima_Setia
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download dataset

Download [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) dari Kaggle dan letakkan di folder `heart_raw/`.

### 4. Jalankan preprocessing

```bash
jupyter notebook preprocessing/preprocessing.ipynb
```

### 5. Tracking eksperimen

```bash
mlflow ui
```

Buka `http://localhost:5000` untuk melihat hasil eksperimen.
