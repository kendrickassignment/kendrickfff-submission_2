# Telco Customer Churn Prediction — MLOps End-to-End System

| | |
|---|---|
| **Nama** | Kendrick Filbert |
| **Username Dicoding** | kendrickfff |
| **Dataset** | [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) (7.032 records) |
| **Kelas** | Machine Learning Operations (MLOps) |

---

## 1. Permasalahan

### Domain: Telekomunikasi — Customer Retention

Customer churn (kehilangan pelanggan) merupakan salah satu tantangan bisnis paling kritis di industri telekomunikasi. Berdasarkan data industri, biaya akuisisi pelanggan baru rata-rata **5-7x lebih mahal** dibanding mempertahankan pelanggan yang sudah ada. Tingkat churn yang tinggi secara langsung menurunkan revenue dan menghambat pertumbuhan bisnis.

### Masalah yang Ingin Diselesaikan

Perusahaan telekomunikasi kesulitan mengidentifikasi pelanggan yang berpotensi churn **sebelum** mereka benar-benar berhenti berlangganan. Tanpa sistem prediksi yang akurat, tim product dan marketing tidak dapat melakukan intervensi proaktif secara tepat sasaran.

### Pendekatan Solusi

Membangun sebuah **end-to-end machine learning system** yang mampu:
1. Memprediksi probabilitas churn setiap pelanggan berdasarkan data profil dan perilaku.
2. Berjalan secara otomatis di cloud sebagai API yang dapat diakses oleh sistem internal.
3. Dipantau performanya secara real-time melalui monitoring dashboard.

---

## 2. Dataset

### Informasi Dataset

| Properti | Detail |
|---|---|
| Nama | [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) |
| Sumber | IBM Developer / Kaggle |
| Jumlah Record | 7.032 (setelah cleaning) |
| Jumlah Fitur | 13 features + 1 label |
| Format | CSV |
| Target | `Churn` (Yes / No) |
| Churn Rate | 26.6% (imbalanced) |

### Deskripsi Fitur

| Fitur | Tipe | Deskripsi |
|---|---|---|
| `customerID` | String | ID unik pelanggan |
| `gender` | Categorical | Jenis kelamin (Male/Female) |
| `SeniorCitizen` | Integer | Senior citizen flag (0/1) |
| `Partner` | Categorical | Memiliki partner (Yes/No) |
| `Dependents` | Categorical | Memiliki tanggungan (Yes/No) |
| `tenure` | Numerical | Lama berlangganan (bulan) |
| `PhoneService` | Categorical | Layanan telepon (Yes/No) |
| `InternetService` | Categorical | Jenis internet (DSL/Fiber optic/No) |
| `Contract` | Categorical | Tipe kontrak (Month-to-month/One year/Two year) |
| `PaperlessBilling` | Categorical | Paperless billing (Yes/No) |
| `PaymentMethod` | Categorical | Metode pembayaran |
| `MonthlyCharges` | Numerical | Biaya bulanan (USD) |
| `TotalCharges` | Numerical | Total biaya kumulatif (USD) |
| **`Churn`** | **Label** | **Target: pelanggan churn (Yes/No)** |

---

## 3. Solusi Machine Learning

### Arsitektur Model

Menggunakan **Deep Neural Network (DNN)** dengan arsitektur berikut:

```
Input Layer
├── Numerical Features → Z-Score Normalization
├── SeniorCitizen → Pass-through 
└── Categorical Features → Vocabulary Encoding → Embedding Layer

Concatenation Layer
↓
Dense(128, relu) + BatchNorm + Dropout(0.3)
↓
Dense(64, relu) + BatchNorm + Dropout(0.3)
↓
Dense(32, relu) + BatchNorm + Dropout(0.3)
↓
Dense(1, sigmoid) → Churn Probability
```

### Feature Engineering (Transform Component)

| Fitur | Transformasi |
|---|---|
| `tenure`, `MonthlyCharges`, `TotalCharges` | Z-score normalization (`tft.scale_to_z_score`) |
| `SeniorCitizen` | Pass-through (sudah 0/1) |
| `gender`, `Partner`, `Dependents`, dll. | Vocabulary encoding (`tft.compute_and_apply_vocabulary`) |
| `Churn` (label) | Binary mapping: Yes→1, No→0 |

### Metrik Evaluasi

| Metrik | Threshold | Target |
|---|---|---|
| **Binary Accuracy** | ≥ 0.78 | Akurasi klasifikasi keseluruhan |
| **AUC** | ≥ 0.75 | Kemampuan model membedakan kelas positif dan negatif |

### Hyperparameter Tuning (Tuner Component) ⭐

Automated hyperparameter search menggunakan **Keras Tuner (RandomSearch)**:

| Parameter | Search Space |
|---|---|
| Jumlah hidden layers | 2–4 |
| Units per layer | 32, 64, 128, 256 |
| Dropout rate | 0.1–0.5 |
| Embedding dimension | 4, 8, 16 |
| Learning rate | 0.01, 0.001, 0.0001 |

---

## 4. Machine Learning Pipeline (TFX)

Pipeline dibangun menggunakan **TensorFlow Extended (TFX) v1.12.0** dengan **Apache Beam** sebagai orchestrator.

### Komponen Pipeline

```
ExampleGen → StatisticsGen → SchemaGen → ExampleValidator
 ↓
 Transform
 ↓
 Tuner ⭐
 ↓
 Trainer
 ↓
 Resolver
 ↓
 Evaluator
 ↓
 Pusher
```

| # | Komponen | Fungsi |
|---|---|---|
| 1 | **ExampleGen** | Ingest CSV, split 80/20 train-eval |
| 2 | **StatisticsGen** | Generate statistik deskriptif |
| 3 | **SchemaGen** | Infer schema dari statistik |
| 4 | **ExampleValidator** | Validasi data terhadap schema |
| 5 | **Transform** | Feature engineering & preprocessing |
| 6 | **Tuner** | Hyperparameter tuning otomatis ⭐ |
| 7 | **Trainer** | Training DNN model |
| 8 | **Resolver** | Resolve baseline model |
| 9 | **Evaluator** | Evaluasi & blessing model |
| 10 | **Pusher** | Push model ke serving directory |

---

## 5. Performa Model

Hasil evaluasi model menggunakan **TFMA (TensorFlow Model Analysis)**:

| Metrik | Nilai | Threshold | Status |
|---|---|---|---|
| Binary Accuracy | 0.808 | 0.78 | ✅ Pass |
| AUC | 0.870 | 0.75 | ✅ Pass |

### Prediction Test Results

| Customer Profile | Churn Probability | Klasifikasi |
|---|---|---|
| High Risk (Month-to-month, tenure 2) | 87.93% | 🔴 CHURN |
| Low Risk (Two year, tenure 60) | 1.20% | 🟢 RETAIN |
| Medium Risk (One year, tenure 24) | 11.43% | 🟢 RETAIN |

> Model menunjukkan kemampuan yang sangat baik dalam membedakan pelanggan berisiko tinggi dan rendah.

### Slicing Analysis

Evaluasi dilakukan dengan slicing berdasarkan:
- **Overall** — performa keseluruhan
- **Contract type** — Month-to-month vs One year vs Two year
- **Internet Service** — DSL vs Fiber optic vs No internet

---

## 6. Deployment

### Platform & Arsitektur

| Komponen | Teknologi |
|---|---|
| Model Serving | TensorFlow Serving 2.11.0 |
| Container | Docker |
| Cloud Platform | Railway |
| API | REST API (port 8501) |

### Deployment Steps

1. Build Docker image dari Dockerfile
2. Push ke Railway via GitHub repository
3. Model diakses via REST API endpoint

### API Endpoint

```
POST https://kendrickfff-submission2-production.up.railway.app/v1/models/churn-model:predict
```

### Tautan Web App

> **Serving URL:** `https://kendrickfff-submission2-production.up.railway.app/v1/models/churn-model`
>
> **GitHub Repository:** `https://github.com/kendrickassignment/kendrickfff-submission_2`

### Contoh Request

```json
{
  "signature_name": "serving_default",
  "instances": [
    {
      "examples": {
        "b64": "<base64-encoded tf.Example>"
      }
    }
  ]
}
```

---

## 7. Monitoring

### Prometheus Monitoring

Sistem machine learning dipantau menggunakan **Prometheus** yang scrape metrics dari TensorFlow Serving.

**Metrics yang dipantau:**
- `:tensorflow:serving:request_count` — Jumlah total request
- `:tensorflow:serving:request_latency` — Latensi per request
- `:tensorflow:serving:request_log_count` — Log count
- Model version & status

### Grafana Dashboard ⭐

Prometheus disinkronkan dengan **Grafana** untuk visualisasi monitoring yang lebih baik:
- Request rate per menit
- Latency percentiles (p50, p95, p99)
- Error rate
- Model version tracking

### Menjalankan Monitoring Stack

```bash
cd monitoring
docker-compose up -d
```

- **Prometheus:** `http://localhost:9090`
- **Grafana:** `http://localhost:3000` (admin/admin)

### Hasil Monitoring

> Lihat screenshot: `kendrickfff-monitoring.png` dan `kendrickfff-grafana-dashboard.png`

---

## 8. Struktur Proyek

```
.
├── README.md                    # Dokumentasi proyek (file ini)
├── kendrickfff-pipeline.ipynb   # Notebook utama pipeline TFX
├── kendrickfff-testing.ipynb    # Notebook testing prediction ⭐
├── requirements.txt             # Dependencies
├── Dockerfile                   # Dockerfile untuk deployment
├── monitoring.config            # TF Serving monitoring config
│
├── data/
│   └── churn.csv                # Dataset
│
├── modules/                     # Modul pipeline (clean code) ⭐
│   ├── __init__.py
│   ├── transform_module.py      # Preprocessing & feature engineering
│   ├── trainer_module.py        # Model definition & training
│   └── tuner_module.py          # Hyperparameter tuning ⭐
│
├── kendrickfff-pipeline/        # Pipeline artifacts
│   ├── metadata/
│   ├── serving_model/
│   └── ...
│
├── monitoring/                  # Monitoring stack
│   ├── Dockerfile               # Prometheus Dockerfile
│   ├── prometheus.yml           # Prometheus config
│   ├── prometheus.config        # TF Serving metrics config
│   ├── docker-compose.yml       # Docker Compose (Prometheus + Grafana)
│   └── provisioning/
│       └── datasources/
│           └── datasource.yml   # Grafana datasource config
│
├── kendrickfff-deployment.png   # Screenshot deployment ✅
├── kendrickfff-monitoring.png   # Screenshot Prometheus ✅
├── kendrickfff-grafana-dashboard.png  # Screenshot Grafana ⭐
└── kendrickfff-pylint.png       # Screenshot pylint score ⭐
```

---

## 9. Cara Menjalankan

### Prerequisites

```bash
python -m venv mlops-env
source mlops-env/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 1. Generate Dataset

```bash
python preprocess_data.py
```

### 2. Jalankan Pipeline

Buka dan jalankan seluruh cell di `kendrickfff-pipeline.ipynb`.

### 3. Build & Deploy ke Railway

```bash
# Build Docker image
docker build -t churn-model-serving .

# Test locally
docker run -p 8501:8501 churn-model-serving

# Deploy ke Railway via GitHub
git push origin main
```

### 4. Jalankan Monitoring

```bash
cd monitoring
docker-compose up -d
```

### 5. Test Predictions

Jalankan seluruh cell di `kendrickfff-testing.ipynb`.

### 6. Pylint Check ⭐

```bash
pylint modules/transform_module.py modules/trainer_module.py modules/tuner_module.py
```

> Score: **8.93/10** ✅

---

## 10. Saran yang Diterapkan

| # | Saran | Status |
|---|---|---|
| 1 | Komponen **Tuner** untuk hyperparameter tuning | ✅ |
| 2 | **Clean code** + modules directory + pylint (8.93/10) | ✅ |
| 3 | **Testing notebook** untuk prediction request | ✅ |
| 4 | **Grafana dashboard** monitoring | ✅ |

---

*Proyek ini dibuat sebagai submission akhir kelas Machine Learning Operations (MLOps) — Dicoding Indonesia.*
